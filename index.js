
const express = require("express");
const cors = require("cors");
const sharp = require("sharp");
const cvModule = require("@techstark/opencv-js");
require("dotenv").config();

const app = express();

app.use(cors());
app.use(express.json({ limit: "1mb" }));

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
      orbAvailable: Boolean(cv.ORB || cv.ORB_create)
    });
  } catch (error) {
    console.error("OpenCV error:", error.message);

    res.status(500).json({
      success: false,
      error: "OpenCV initialization failed"
    });
  }
});

async function loadImageFromUrl(url) {
  const allowedHost = process.env.SUPABASE_HOST;

  if (!allowedHost) {
    throw new Error("SUPABASE_HOST environment variable is missing");
  }

  const parsed = new URL(url);

  if (parsed.protocol !== "https:" || parsed.hostname !== allowedHost) {
    throw new Error("Image URL must use the configured Supabase host");
  }

  const response = await fetch(url, {
    signal: AbortSignal.timeout(10000)
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
  const mat = new cv.Mat(image.height, image.width, cv.CV_8UC1);
  mat.data.set(image.data);
  return mat;
}

app.post("/compare", async (req, res) => {
  let matA;
  let matB;
  let descriptorsA;
  let descriptorsB;
  let keypointsA;
  let keypointsB;
  let orb;
  let matcher;
  let matches;

  try {
    const { imageUrlA, imageUrlB } = req.body || {};

    if (!imageUrlA || !imageUrlB) {
      return res.status(400).json({
        success: false,
        error: "imageUrlA and imageUrlB are required"
      });
    }

    const cv = await getCV();

    if (!cv.ORB || !cv.BFMatcher) {
      throw new Error("ORB/BFMatcher is not available in this OpenCV build");
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

    orb = cv.ORB.create(500);

    orb.detectAndCompute(
      matA, new cv.Mat(), keypointsA, descriptorsA
    );

    orb.detectAndCompute(
      matB, new cv.Mat(), keypointsB, descriptorsB
    );

    if (descriptorsA.empty() || descriptorsB.empty()) {
      return res.json({
        success: true,
        similar: false,
        score: 0,
        message: "Insufficient image features"
      });
    }

    matcher = new cv.BFMatcher(cv.NORM_HAMMING, true);
    matches = new cv.DMatchVector();

    matcher.match(descriptorsA, descriptorsB, matches);

    let goodMatches = 0;

    for (let i = 0; i < matches.size(); i++) {
      if (matches.get(i).distance < 50) goodMatches++;
    }

    const total = Math.max(
      1,
      Math.min(keypointsA.size(), keypointsB.size())
    );

    const score = Math.round((goodMatches / total) * 100);

    res.json({
      success: true,
      similar: goodMatches >= 12 && score >= 10,
      score: Math.min(score, 100),
      goodMatches
    });
  } catch (error) {
    console.error("Comparison error:", error.message);

    res.status(500).json({
      success: false,
      error: error.message
    });
  } finally {
    [
      matA, matB, descriptorsA, descriptorsB,
      keypointsA, keypointsB, orb, matcher, matches
    ].forEach((item) => {
      if (item && typeof item.delete === "function") {
        item.delete();
      }
    });
  }
});

module.exports = app;
