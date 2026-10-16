
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
const supabaseHost = process.env.SUPABASE_HOST;

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

async function getCV() {
  if (cvPromise) return cvPromise;

  cvPromise = (async () => {
    if (cvModule.Mat && cvModule.ORB) return cvModule;

    if (typeof cvModule.then === "function") {
      return await cvModule;
    }

    await new Promise((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error("OpenCV initialization timed out")),
        15000
      );

      cvModule.onRuntimeInitialized = () => {
        clearTimeout(timeout);
        resolve();
      };
    });

    return cvModule;
  })();

  return cvPromise;
}

app.get("/", (req, res) => {
  res.json({
    success: true,
    service: "HosPro Duplicate Complaints API"
  });
});

app.get("/health", async (req, res) => {
  try {
    const cv = await getCV();

    res.json({
      success: true,
      service: "HosPro Duplicate Complaints API",
      openCVLoaded: Boolean(cv.Mat),
      orbAvailable: Boolean(cv.ORB || cv.ORB_create),
      supabaseConfigured: Boolean(supabase)
    });
  } catch (error) {
    console.error("Health error:", error.message);

    res.status(500).json({
      success: false,
      error: "Service initialization failed"
    });
  }
});

async function loadImageFromUrl(url) {
  if (!supabaseHost) {
    throw new Error("SUPABASE_HOST is not configured");
  }

  const parsed = new URL(url);

  if (
    parsed.protocol !== "https:" ||
    parsed.hostname !== supabaseHost ||
    parsed.username ||
    parsed.password
  ) {
    throw new Error("Image URL is not from the configured Supabase host");
  }

  const response = await fetch(url, {
    signal: AbortSignal.timeout(10000),
    redirect: "error"
  });

  if (!response.ok) {
    throw new Error("Unable to download image");
  }

  const contentType = response.headers.get("content-type") || "";

  if (!contentType.startsWith("image/")) {
    throw new Error("URL did not return an image");
  }

  const buffer = Buffer.from(await response.arrayBuffer());

  if (buffer.length > 8 * 1024 * 1024) {
    throw new Error("Image must be smaller than 8 MB");
  }

  const result = await sharp(buffer)
    .rotate()
    .resize(640, 640, { fit: "inside" })
    .greyscale()
    .raw()
    .toBuffer({ resolveWithObject: true });

  return {
    data: result.data,
    width: result.info.width,
    height: result.info.height
  };
}

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

    if (!cv.ORB || !cv.BFMatcher) {
      throw new Error("ORB/BFMatcher is unavailable");
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

    orb = cv.ORB.create(500);

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
      matA, matB, descriptorsA, descriptorsB,
      keypointsA, keypointsB, orb, matcher, matches,
      emptyMask
    ].forEach((item) => {
      if (item && typeof item.delete === "function") {
        item.delete();
      }
    });
  }
}

// Existing single-pair comparison endpoint
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

    res.json({
      success: true,
      ...result
    });
  } catch (error) {
    console.error("Comparison error:", error.message);

    res.status(500).json({
      success: false,
      error: "Image comparison failed"
    });
  }
});

// New: scan complaints for possible duplicates
app.get("/duplicate-complaints", async (req, res) => {
  if (!supabase) {
    return res.status(500).json({
      success: false,
      error: "Supabase environment variables are missing"
    });
  }

  try {
    // Limit keeps this first version manageable on Vercel.
    const { data: complaints, error } = await supabase
      .from("complaints")
      .select(
        "id, name, hostel_name, room_number, complaint_description, image_url, status, created_at, username"
      )
      .order("created_at", { ascending: false })
      .limit(200);

    if (error) throw error;

    const groups = new Map();

    for (const complaint of complaints || []) {
      if (!complaint.hostel_name || !complaint.room_number) {
        continue;
      }

      if (!complaint.image_url) continue;

      let imageHost;

      try {
        imageHost = new URL(complaint.image_url).hostname;
      } catch {
        continue;
      }

      // Skip test URLs and any URL outside our Supabase host.
      if (
        !supabaseHost ||
        imageHost !== supabaseHost ||
        !complaint.image_url.startsWith("https://")
      ) {
        continue;
      }

      const key = [
        complaint.hostel_name.trim().toLowerCase(),
        complaint.room_number.trim().toLowerCase()
      ].join("::");

      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(complaint);
    }

    const duplicates = [];
    const errors = [];
    let comparisons = 0;
    const MAX_COMPARISONS = 100;

    for (const roomComplaints of groups.values()) {
      if (roomComplaints.length < 2) continue;

      for (let i = 0; i < roomComplaints.length; i++) {
        for (let j = i + 1; j < roomComplaints.length; j++) {
          if (comparisons >= MAX_COMPARISONS) break;

          const a = roomComplaints[i];
          const b = roomComplaints[j];

          // Avoid comparing the same stored image URL.
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
            console.error(
              `Could not compare complaints ${a.id} and ${b.id}:`,
              error.message
            );

            errors.push({
              complaintIds: [a.id, b.id],
              error: "Image comparison failed"
            });
          }
        }

        if (comparisons >= MAX_COMPARISONS) break;
      }

      if (comparisons >= MAX_COMPARISONS) break;
    }

    res.json({
      success: true,
      scannedComplaints: (complaints || []).length,
      comparisons,
      comparisonLimitReached: comparisons >= MAX_COMPARISONS,
      duplicateCount: duplicates.length,
      duplicates,
      failedComparisons: errors.length,
      message:
        "Results are possible duplicates, not confirmed duplicates."
    });
  } catch (error) {
    console.error("Duplicate scan error:", error.message);

    res.status(500).json({
      success: false,
      error: "Could not scan complaints"
    });
  }
});

module.exports = app;
