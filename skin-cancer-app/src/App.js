import React, { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";

function App() {
  const [model, setModel] = useState(null);
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);

  // Labels (tumhare dataset ke diagnosis ke codes)
  const labels = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];

  // Model load karo
  useEffect(() => {
  const loadModel = async () => {
    try {
      const loadedModel = await tf.loadLayersModel("/tfjs_model/model.json");

      // 👇 Agar training 224x224x3 pe hui thi toh ye sahi hai
      loadedModel.build([null, 224, 224, 3]);

      setModel(loadedModel);
      console.log("✅ Model loaded successfully");
      console.log("Model Input Shape:", loadedModel.inputs[0].shape); // <-- yahan change

    } catch (error) {
      console.error("❌ Error loading model:", error);
    }
  };
  loadModel();
}, []);


  // Image handle karo
  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setImage(URL.createObjectURL(file));
      setPrediction(null);
    }
  };

  // Predict function
  const handlePredict = async () => {
  if (!model || !image) return;

  const img = document.getElementById("uploadedImage");

  // 🔹 Model input shape se size lo (e.g. [null, 224, 224, 3])
  const inputShape = model.inputs[0].shape;
  const targetH = inputShape[1];
  const targetW = inputShape[2];

  console.log("📏 Model expects:", targetH, "x", targetW);

  let tensor = tf.browser.fromPixels(img)
    .resizeNearestNeighbor([targetH, targetW])
    .toFloat()
    .div(tf.scalar(255.0))
    .expandDims();

  console.log("🔍 Tensor shape:", tensor.shape);

  const predictions = await model.predict(tensor).data();
  console.log("📊 Raw Predictions:", predictions);

  const maxIndex = predictions.indexOf(Math.max(...predictions));

  setPrediction({
    label: labels[maxIndex],
    confidence: (predictions[maxIndex] * 100).toFixed(2) + "%"
  });
};

  return (
    <div style={{ textAlign: "center", marginTop: "30px" }}>
      <h1>Skin Cancer Classification</h1>

      <input type="file" accept="image/*" onChange={handleImageUpload} />

      {image && (
        <div>
          <img
            id="uploadedImage"
            src={image}
            alt="Uploaded"
            width="224"
            height="224"
            style={{ marginTop: "20px", border: "2px solid #ccc" }}
          />
          <br />
          <button onClick={handlePredict} style={{ marginTop: "20px" }}>
            Predict
          </button>
        </div>
      )}

      {prediction && (
        <div style={{ marginTop: "20px" }}>
          <h2>Prediction: {prediction.label}</h2>
          <p>Confidence: {prediction.confidence}</p>
        </div>
      )}
    </div>
  );
}

export default App;
