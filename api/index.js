
const express = require("express");
const cors = require("cors");
const sharp = require("sharp");
const { createClient } = require("@supabase/supabase-js");
const cvModule = require("@techstark/opencv-js");
require("dotenv").config();

const app = express();

app.use(cors());
app.use(express.json({ limit: "1mb" }));

const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
const supabaseHost = (process.env.SUPABASE_HOST || "")
  .trim()
  .replace(/^https?:\/\//, "")
  .replace(/\/+$/, "");

const supabase =
  supabaseUrl && supabaseKey
    ? createClient(supabaseUrl, supabaseKey, {
        auth: {
          persistSession: false,
          autoRefreshToken: false
        }
      })
    : null;

let cvPromise;

/* =========================================================
   BASIC ROUTES
========================================================= */

app.get("/", (req, res) => {
  res.status(200).json({
    success: true,
    service: "HosPro Duplicate Complaints API",
    message: "API is running"
  });
});

// Temporary route to verify that this exact file is deployed.
app.get("/debug-route-check", (req, res) => {
  res.status(200).json({
    success: true,
    source: "api/index.js",
    message: "Correct Express function is running"
  });
});

app.get("/health", async (req, res) => {
  res.status(200).json({
    success: true,
    service: "HosPro Duplicate Complaints API",
    opencvPackageLoaded: Boolean(cvModule),
    supabaseConfigured: Boolean(supabase),
    supabaseHostConfigured: Boolean(supabaseHost)
  });
});

/* =========================================================
   OPENCV INITIALIZATION
========================================================= */

async function getCV() {
  if (cvPromise) return cvPromise;

  cvPromise = (async () => {
    let cv = cvModule;

    if (cv && typeof cv.then === "function") {
      cv = await cv;
    }

    if (cv && cv.Mat && (cv.ORB || cv.ORB_create)) {
      return cv;
    }

    if (!cv) {
      throw new Error("OpenCV module could not be loaded");
    }

    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error("OpenCV initialization timed out"));
      }, 15000);

      cv.onRuntimeInitialized = () => {
        clearTimeout(timeout);
        resolve();
      };
    });

    return cv;
  })().catch((error) => {
    // Permit a later request to retry initialization.
    cvPromise = null;
    throw error;
  });

  return cvPromise;
}

/* =========================================================
   IMAGE DOWNLOAD AND VALIDATION
========================================================= */

async function loadImageFromUrl(imageUrl) {
  if (!supabaseHost) {
    throw new Error("SUPABASE_HOST is not configured");
  }

  let parsedUrl;

  try {
    parsedUrl = new URL(imageUrl);
  } catch {
    throw new Error("Invalid image URL");
  }

  if (
    parsedUrl.protocol !== "https:" ||
    parsedUrl.hostname !== supabaseHost ||
    parsedUrl.username ||
    parsedUrl.password
  ) {
    throw new Error("Image URL host does not match SUPABASE_HOST");
  }

  const response = await fetch(parsedUrl.toString(), {
    signal: AbortSignal.timeout(10000),
    redirect: "error"
  });

  if (!response.ok) {
    throw new Error(`Image download failed: HTTP ${response.status}`);
  }

  const contentType = response.headers.get("content-type") || "";

  if (!contentType.toLowerCase().startsWith("image/")) {
    throw new Error("URL did not return an image");
  }

  // Read at most 8 MB plus one byte to enforce the limit.
  const reader = response.body.getReader();
  const chunks = [];
  let totalBytes = 0;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      totalBytes += value.length;

      if (totalBytes > 8 * 1024 * 1024) {
        await reader.cancel();
        throw new Error("Image must be smaller than 8 MB");
      }

      chunks.push(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
  }

  const buffer = Buffer.concat(chunks);

  const processed = await sharp(buffer)
    .rotate()
    .resize(640, 640, { fit: "inside" })
    .greyscale()
    .raw()
    .toBuffer({ resolveWithObject: true });

  return {
    data: processed.data,
    width: processed.info.width,
    height: processed.info.height
  };
}

/* =========================================================
   OPENCV HELPERS
========================================================= */

function makeMat(cv, image) {
  const mat = new cv.Mat(
    image.height,
    image.width,
    cv.CV_8UC1
  );

  mat.data.set(image.data);
  return mat;
}

async function compareImageUrls(imageUrlA, imageUrlB) {
  let matA;
  let matB;
  let descriptorsA;
  let descriptorsB;
  let keypointsA;
  let keypointsB;
  let orb;
  let matcher;
  let matches;
  let emptyMask;

  try {
    const cv = await getCV();

    const ORB = cv.ORB || cv.ORB_create;

    if (!ORB || !cv.BFMatcher) {
      throw new Error("OpenCV ORB/BFMatcher is unavailable");
    }

    const [imageA, imageB] = await Promise.all([
      loadImageFromUrl(imageUrlA),
      loadImageFromUrl(imageUrlB)
    ]);

    matA = makeMat(cv, imageA);
    matB = makeMat(cv, imageB);

    descriptorsA = new cv.Mat();
    descriptorsB = new cv.Mat();
    keypointsA = new cv.KeyPointVector();
    keypointsB = new cv.KeyPointVector();
    emptyMask = new cv.Mat();

    orb = typeof ORB.create === "function"
      ? ORB.create(500)
      : new ORB(500);

    orb.detectAndCompute(
      matA, emptyMask, keypointsA, descriptorsA
    );

    orb.detectAndCompute(
      matB, emptyMask, keypointsB, descriptorsB
    );

    if (
      descriptorsA.empty() ||
      descriptorsB.empty() ||
      keypointsA.size() === 0 ||
      keypointsB.size() === 0
    ) {
      return {
        similar: false,
        score: 0,
        goodMatches: 0
      };
    }

    matcher = new cv.BFMatcher(cv.NORM_HAMMING, true);
    matches = new cv.DMatchVector();

    matcher.match(descriptorsA, descriptorsB, matches);

    let goodMatches = 0;

    for (let i = 0; i < matches.size(); i++) {
      if (matches.get(i).distance < 50) {
        goodMatches++;
      }
    }

    const total = Math.max(
      1,
      Math.min(keypointsA.size(), keypointsB.size())
    );

    const score = Math.min(
      100,
      Math.round((goodMatches / total) * 100)
    );

    return {
      similar: goodMatches >= 12 && score >= 10,
      score,
      goodMatches
    };
  } finally {
    [
      matA, matB,
      descriptorsA, descriptorsB,
      keypointsA, keypointsB,
      orb, matcher, matches, emptyMask
    ].forEach((item) => {
      if (item && typeof item.delete === "function") {
        try {
          item.delete();
        } catch (_) {
          // Ignore native/WASM cleanup errors.
        }
      }
    });
  }
}

/* =========================================================
   COMPARE TWO IMAGES
========================================================= */

app.post("/compare", async (req, res) => {
  try {
    const { imageUrlA, imageUrlB } = req.body || {};

    if (!imageUrlA || !imageUrlB) {
      return res.status(400).json({
        success: false,
        error: "imageUrlA and imageUrlB are required"
      });
    }

    const result = await compareImageUrls(
      imageUrlA,
      imageUrlB
    );

    return res.json({
      success: true,
      ...result
    });
  } catch (error) {
    console.error("Comparison error:", error.message);

    return res.status(500).json({
      success: false,
      error: "Image comparison failed",
      details: error.message
    });
  }
});

/* =========================================================
   FIND POSSIBLE DUPLICATE COMPLAINTS
========================================================= */

app.get("/duplicate-complaints", async (req, res) => {
  if (!supabase) {
    return res.status(500).json({
      success: false,
      error: "Supabase environment variables are missing",
      supabaseUrlConfigured: Boolean(supabaseUrl),
      serviceKeyConfigured: Boolean(supabaseKey)
    });
  }

  if (!supabaseHost) {
    return res.status(500).json({
      success: false,
      error: "SUPABASE_HOST is not configured"
    });
  }

  try {
    const { data: complaints, error } = await supabase
      .from("complaints")
      .select(
        "id, name, hostel_name, room_number, complaint_description, image_url, status, created_at, username"
      )
      .order("created_at", { ascending: false })
      .limit(200);

    if (error) throw error;

    const groups = new Map();

    let missingLocation = 0;
    let missingImage = 0;
    let invalidImageHost = 0;
    let validImages = 0;

    for (const complaint of complaints || []) {
      if (!complaint.hostel_name || !complaint.room_number) {
        missingLocation++;
        continue;
      }

      if (!complaint.image_url) {
        missingImage++;
        continue;
      }

      let imageUrl;

      try {
        imageUrl = new URL(complaint.image_url);
      } catch {
        invalidImageHost++;
        continue;
      }

      if (
        imageUrl.protocol !== "https:" ||
        imageUrl.hostname !== supabaseHost
      ) {
        invalidImageHost++;
        continue;
      }

      validImages++;

      const key = [
        complaint.hostel_name.trim().toLowerCase(),
        complaint.room_number.trim().toLowerCase()
      ].join("::");

      if (!groups.has(key)) {
        groups.set(key, []);
      }

      groups.get(key).push(complaint);
    }

    const duplicates = [];
    let comparisons = 0;
    let failedComparisons = 0;
    let comparisonLimitReached = false;

    const MAX_COMPARISONS = 100;

    for (const roomComplaints of groups.values()) {
      if (roomComplaints.length < 2) continue;

      for (let i = 0; i < roomComplaints.length; i++) {
        for (let j = i + 1; j < roomComplaints.length; j++) {
          if (comparisons >= MAX_COMPARISONS) {
            comparisonLimitReached = true;
            break;
          }

          const a = roomComplaints[i];
          const b = roomComplaints[j];

          // Identical stored URLs need no image comparison.
          if (a.image_url === b.image_url) continue;

          comparisons++;

          try {
            const result = await compareImageUrls(
              a.image_url,
              b.image_url
            );

            if (result.similar) {
              duplicates.push({
                hostel_name: a.hostel_name,
                room_number: a.room_number,
                score: result.score,
                goodMatches: result.goodMatches,
                complaintA: a,
                complaintB: b
              });
            }
          } catch (error) {
            failedComparisons++;

            console.error(
              `Comparison failed for complaint IDs ${a.id}, ${b.id}:`,
              error.message
            );
          }
        }

        if (comparisons >= MAX_COMPARISONS) {
          comparisonLimitReached = true;
          break;
        }
      }

      if (comparisons >= MAX_COMPARISONS) {
        comparisonLimitReached = true;
        break;
      }
    }

    return res.status(200).json({
      success: true,
      scannedComplaints: (complaints || []).length,
      validImages,
      missingLocation,
      missingImage,
      invalidImageHost,
      comparisons,
      comparisonLimitReached,
      duplicateCount: duplicates.length,
      duplicates,
      failedComparisons,
      message: comparisons === 0
        ? "No image pairs were compared. Check image URLs and hostel/room grouping."
        : "Results are possible duplicates, not confirmed duplicates."
    });
  } catch (error) {
    console.error("Duplicate scan error:", error.message);

    return res.status(500).json({
      success: false,
      error: "Could not scan complaints",
      details: error.message
    });
  }
});

/* =========================================================
   EXPORT FOR VERCEL
========================================================= */

module.exports = app;
