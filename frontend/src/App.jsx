
import { useState } from "react";

import {
  Brain,
  LayoutDashboard,
  Scan,
  Layers,
  Info,
  Settings,
  Bell,
  UserRound
} from "lucide-react";

import "./App.css";

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [activePage, setActivePage] = useState("Dashboard");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState("");
  const [prediction, setPrediction] = useState("");

  // Handle MRI image upload
  const handleImageUpload = (e) => {
    const file = e.target.files[0];

    if (!file) return;

    if (!file.type.startsWith("image/")) {
      alert("Please select a valid image file.");
      e.target.value = "";
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      alert("Image size must be less than 5 MB.");
      e.target.value = "";
      return;
    }

    if (selectedImage) {
      URL.revokeObjectURL(selectedImage);
    }

    setSelectedFile(file);
    setSelectedImage(URL.createObjectURL(file));
    setPrediction("");
    setAnalysisResult("");
  };

  // Clear uploaded image
  const handleClearImage = () => {
    if (selectedImage) {
      URL.revokeObjectURL(selectedImage);
    }

    setSelectedImage(null);
    setSelectedFile(null);
    setPrediction("");
    setAnalysisResult("");
  };

  // Connect React with Flask backend
  const handleAnalyze = async () => {
    if (!selectedFile) {
      setAnalysisResult("Please upload an MRI image first.");
      return;
    }

    setIsAnalyzing(true);
    setAnalysisResult("");
    setPrediction("");

    try {
      const formData = new FormData();

      formData.append("file", selectedFile);

      const response = await fetch(
        "http://127.0.0.1:5000/api/predict",
        {
          method: "POST",
          body: formData
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Prediction failed.");
      }

      setPrediction(data.prediction);
      setAnalysisResult(data.message);

    } catch (error) {
      setAnalysisResult(
        "Error: " + error.message
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="app-container">

      {/* Sidebar */}
      <aside className="sidebar">

        <div className="logo">
          <Brain size={32} />
          <h2>NeuroScan AI</h2>
        </div>

        <p className="menu-title">MAIN MENU</p>

        <nav>

          <a
            href="#"
            className={`nav-link ${
              activePage === "Dashboard" ? "active" : ""
            }`}
            onClick={(e) => {
              e.preventDefault();
              setActivePage("Dashboard");
            }}
          >
            <LayoutDashboard size={20} />
            Dashboard
          </a>

          <a
            href="#"
            className={`nav-link ${
              activePage === "Tumor Detection" ? "active" : ""
            }`}
            onClick={(e) => {
              e.preventDefault();
              setActivePage("Tumor Detection");
            }}
          >
            <Scan size={20} />
            Tumor Detection
          </a>

          <a
            href="#"
            className={`nav-link ${
              activePage === "Segmentation" ? "active" : ""
            }`}
            onClick={(e) => {
              e.preventDefault();
              setActivePage("Segmentation");
            }}
          >
            <Layers size={20} />
            Segmentation
          </a>

          <a
            href="#"
            className={`nav-link ${
              activePage === "About Project" ? "active" : ""
            }`}
            onClick={(e) => {
              e.preventDefault();
              setActivePage("About Project");
            }}
          >
            <Info size={20} />
            About Project
          </a>

        </nav>

        <div className="sidebar-bottom">
          <Settings size={20} />
          <span>Settings</span>
        </div>

      </aside>

      {/* Main Content */}
      <main className="main-content">

        {/* Header */}
        <header className="top-header">

          <div className="header-title">
            <h3>Brain Tumor Detection System</h3>
            <span>AI Medical Dashboard</span>
          </div>

          <div className="header-profile">

            <Bell size={21} />

            <div className="profile-icon">
              <UserRound size={22} />
            </div>

            <div className="profile-info">
              <h4>Admin</h4>
              <p>Project User</p>
            </div>

          </div>

        </header>

        <section className="welcome-section">

          {/* Dashboard */}
          {activePage === "Dashboard" && (
            <>
              <h1>Welcome to NeuroScan AI</h1>

              <p>
                MRI Brain Tumor Detection and Segmentation
              </p>

              <div className="dashboard-cards">

                <div
                  className="dashboard-card"
                  onClick={() => setActivePage("Tumor Detection")}
                >
                  <Brain size={32} />
                  <h3>Brain MRI Analysis</h3>
                  <p>Upload and explore MRI scans</p>
                </div>

                <div
                  className="dashboard-card"
                  onClick={() => setActivePage("Tumor Detection")}
                >
                  <Scan size={32} />
                  <h3>Tumor Detection</h3>
                  <p>4 classification categories</p>
                </div>

                <div
                  className="dashboard-card"
                  onClick={() => setActivePage("Segmentation")}
                >
                  <Layers size={32} />
                  <h3>Segmentation</h3>
                  <p>Visualize tumor regions</p>
                </div>

              </div>

              <div className="dashboard-info">
                <h2>System Overview</h2>

                <p>
                  NeuroScan AI provides an interface for MRI image
                  upload, brain tumor classification and
                  segmentation visualization.
                </p>
              </div>
            </>
          )}

          {/* Tumor Detection */}
          {activePage === "Tumor Detection" && (
            <>
              <h1>Tumor Detection</h1>

              <p>
                Upload an MRI brain image to begin tumor classification.
              </p>

              <div className="upload-container">

                <Scan size={45} />

                <h2>Upload MRI Image</h2>

                <p>
                  Select an MRI brain scan image from your computer.
                </p>

                <input
                  type="file"
                  accept=".jpg,.jpeg,.png,.bmp"
                  onChange={handleImageUpload}
                />

                <p className="upload-hint">
                  Supported: JPG, JPEG, PNG, BMP |
                  Maximum size: 5 MB
                </p>

                {/* Image Preview */}
                {selectedImage && (
                  <div className="image-preview">

                    <h3>Uploaded MRI Image</h3>

                    <img
                      src={selectedImage}
                      alt="Uploaded MRI Scan"
                      width="300"
                    />

                    <button
                      className="clear-button"
                      onClick={handleClearImage}
                      disabled={isAnalyzing}
                    >
                      Clear Image
                    </button>

                    <button
                      className="analyze-button"
                      disabled={isAnalyzing}
                      onClick={handleAnalyze}
                    >
                      {isAnalyzing
                        ? "Analyzing MRI..."
                        : "Start Analysis"}
                    </button>

                    {isAnalyzing && (
                      <p className="analysis-status">
                        Processing MRI image using SVM model...
                      </p>
                    )}

                    {analysisResult && (
                      <div className="analysis-status">
                        <Info size={20} />
                        <p>{analysisResult}</p>
                      </div>
                    )}

                  </div>
                )}

                {/* Classification Result */}
                {selectedImage && (
                  <div className="classification-result">

                    <h2>Classification Result</h2>

                    {!prediction ? (
                      <>
                        <p>
                          Upload your MRI image and click
                          Start Analysis to get the prediction.
                        </p>

                        <div className="result-details">
                          <h3>Available Tumor Categories</h3>

                          <div className="result-item">Glioma</div>
                          <div className="result-item">Meningioma</div>
                          <div className="result-item">Pituitary</div>
                          <div className="result-item">No Tumor</div>
                        </div>
                      </>
                    ) : (
                      <>
                        <h3>Predicted Category</h3>

                        <div className="result-item">
                          {prediction
                            .replace("_", " ")
                            .toUpperCase()}
                        </div>

                        <p className="result-note">
                          SVM model prediction completed successfully.
                        </p>

                        <p className="result-note">
                          This is an educational model prediction,
                          not a medical diagnosis.
                        </p>
                      </>
                    )}

                  </div>
                )}

              </div>
            </>
          )}

          {/* Segmentation */}
          {activePage === "Segmentation" && (
            <>
              <h1>Brain Tumor Segmentation</h1>

              <p>
                Compare the original MRI image with its
                segmentation output.
              </p>

              <div className="segmentation-container">

                <div className="segmentation-card">
                  <h3>Original MRI Image</h3>

                  {selectedImage ? (
                    <img
                      src={selectedImage}
                      alt="Original MRI"
                    />
                  ) : (
                    <p>
                      Please upload an MRI image in
                      Tumor Detection first.
                    </p>
                  )}
                </div>

                <div className="segmentation-card">
                  <h3>Segmentation Result</h3>

                  <div className="segmentation-placeholder">
                    <Layers size={45} />

                    <p>
                      Segmentation output will appear here.
                    </p>

                    <span>
                      Demo placeholder – model not connected
                    </span>
                  </div>
                </div>

              </div>
            </>
          )}

          {/* About Project */}
          {activePage === "About Project" && (
            <>
              <h1>About NeuroScan AI</h1>

              <p>
                An MRI-based Brain Tumor Detection and
                Segmentation System.
              </p>

              <div className="about-container">

                <div className="about-card">
                  <Brain size={35} />
                  <h3>Project Overview</h3>

                  <p>
                    NeuroScan AI is designed to support
                    brain MRI image analysis through tumor
                    classification and segmentation.
                  </p>
                </div>

                <div className="about-card">
                  <Scan size={35} />
                  <h3>Tumor Detection</h3>

                  <p>
                    The classification module identifies
                    Glioma, Meningioma, Pituitary and
                    No Tumor categories.
                  </p>
                </div>

                <div className="about-card">
                  <Layers size={35} />
                  <h3>Image Segmentation</h3>

                  <p>
                    The segmentation module is designed
                    to highlight tumor regions in MRI images.
                  </p>
                </div>

                <div className="about-card">
                  <Info size={35} />
                  <h3>Project Objective</h3>

                  <p>
                    To provide a simple interface for
                    exploring MRI-based brain tumor analysis.
                  </p>
                </div>

              </div>
            </>
          )}

        </section>

      </main>

    </div>
  );
}

export default App;
