import React, { useState } from "react";
import axios from "axios";
import Header from "./Header"; 
import MiddleSection from "./MiddleSection";
import Footer from "./Footer";



import "@fontsource/inter";
import "./App.css";

function App() {
  const [emailContent, setEmailContent] = useState("");
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post("http://127.0.0.1:5000/analyze", {
        email_content: emailContent,
      });
      setResult(response.data);
    } catch (error) {
      console.error("Error analyzing the email:", error);
      setResult({ error: "Failed to analyze email." });
    }
  };

  return (
    
    <div className="container">
    <div className="css-selector">

      <Header
        title="AI-Powered Phishing Detection"
        subtitle="Analyze email content to detect potential phishing attacks."
      />
      <MiddleSection />
      <Footer />
    </div>
    </div>
  );
}

export default App;
