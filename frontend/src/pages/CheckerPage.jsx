import React, { useState, useRef } from "react";
import { useNavigate, Link } from "react-router-dom";
import { Button } from "../components/ui/button";
import { Textarea } from "../components/ui/textarea";
import { Card } from "../components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "../components/ui/tabs";
import { ArrowLeft, Upload, FileText, Sparkles } from "lucide-react";
import { checkPlagiarism } from "../lib/plagiarism-checker";
import Header from "../components/Header";
import { useAuth } from "../lib/AuthContext";

const SAMPLE_TEXTS = [
  {
    title: "Academic Essay Sample",
    text: "The impact of climate change on global ecosystems has been a subject of intense scientific scrutiny in recent decades. Rising temperatures, shifting precipitation patterns, and extreme weather events are fundamentally altering habitats and threatening biodiversity worldwide. Scientists have documented significant changes in species distribution, with many organisms migrating toward cooler regions or higher elevations. The interconnected nature of ecosystems means that changes in one species can have cascading effects throughout the food web, potentially leading to ecosystem collapse in vulnerable areas.",
  },
  {
    title: "Technology Article Sample",
    text: "Artificial intelligence continues to revolutionize industries across the globe, from healthcare to finance. Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions with remarkable accuracy. Natural language processing has enabled computers to understand and generate human-like text, powering virtual assistants and chatbots. As AI technology advances, questions about ethics, privacy, and the future of work become increasingly important for society to address.",
  },
  {
    title: "Business Report Sample",
    text: "The global marketplace has undergone significant transformation in the digital age. E-commerce platforms have democratized access to international markets, allowing small businesses to compete on a global scale. Supply chain optimization through advanced analytics has reduced costs and improved efficiency. Companies that embrace digital transformation and data-driven decision-making are better positioned to adapt to changing market conditions and customer preferences.",
  },
];

const CheckerPage = () => {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  const [text, setText] = useState("");
  const [isChecking, setIsChecking] = useState(false);
  const [wordCount, setWordCount] = useState(0);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const { token } = useAuth();

  const handleTextChange = (e) => {
    const newText = e.target.value;
    setText(newText);
    const words = newText
      .trim()
      .split(/\s+/)
      .filter((word) => word.length > 0);
    setWordCount(words.length);
  };

  const handleFileChange = async (e) => {
    console.log("handleFileChange called", e.target.files);
    const file = e.target.files?.[0];
    if (!file) {
      console.log("No file selected");
      return;
    }

    console.log("File selected:", file.name, file.type);
    setSelectedFile(file);

    const validTypes = [
      "text/plain",
      "application/pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "application/msword",
    ];

    if (!validTypes.includes(file.type)) {
      console.log("Invalid file type:", file.type);
      alert("Please upload a valid file (TXT, PDF, DOC, or DOCX)");
      setSelectedFile(null);
      return;
    }

    try {
      if (file.type === "text/plain") {
        const text = await file.text();
        setText(text);
        const words = text
          .trim()
          .split(/\s+/)
          .filter((word) => word.length > 0);
        setWordCount(words.length);
        console.log("Text file processed, word count:", words.length);
      } else {
        setText(`File uploaded: ${file.name}\n\nNote: PDF and DOC file text extraction would be implemented with proper libraries.`);
        setWordCount(0);
        console.log("Non-text file uploaded:", file.name);
      }
    } catch (error) {
      console.error("Error reading file:", error);
      alert("Error reading file. Please try again.");
      setSelectedFile(null);
    }
  };

  const handleFileButtonClick = (e) => {
    console.log("Button clicked, fileInputRef:", fileInputRef.current);
    if (e) {
      e.stopPropagation();
    }
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const file = e.dataTransfer.files?.[0];
    if (!file) return;

    setSelectedFile(file);

    const validTypes = [
      "text/plain",
      "application/pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "application/msword",
    ];

    if (!validTypes.includes(file.type)) {
      alert("Please upload a valid file (TXT, PDF, DOC, or DOCX)");
      setSelectedFile(null);
      return;
    }

    try {
      if (file.type === "text/plain") {
        const textContent = await file.text();
        setText(textContent);
        const words = textContent
          .trim()
          .split(/\s+/)
          .filter((word) => word.length > 0);
        setWordCount(words.length);
      } else {
        setText(`File uploaded: ${file.name}\n\nNote: PDF and DOC file text extraction would be implemented with proper libraries.`);
        setWordCount(0);
      }
    } catch (error) {
      console.error("Error reading file:", error);
      alert("Error reading file. Please try again.");
      setSelectedFile(null);
    }
  };

  const handleSampleClick = (sampleText) => {
    setText(sampleText);
    const words = sampleText
      .trim()
      .split(/\s+/)
      .filter((word) => word.length > 0);
    setWordCount(words.length);
  };

  const handleCheck = async () => {
    if (!text.trim()) {
      alert("Please enter some text to check for plagiarism");
      return;
    }

    setIsChecking(true);

    try {
      // Call Flask API instead of local function
      const response = await fetch('http://localhost:5000/api/check-plagiarism', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const result = await response.json();

      // Attempt to save history + report if authenticated
      try {
        if (token) {
          const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5001';
          await fetch(`${API_BASE}/api/history`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${token}`,
            },
            body: JSON.stringify({ text, result, inputType: 'text' }),
          });
        }
      } catch (e) {
        console.warn('Failed to save history/report:', e);
      }

      navigate("/results", { state: { result } });
    } catch (error) {
      console.error("Error checking plagiarism:", error);
      alert("An error occurred while checking for plagiarism. Please make sure the API server is running (python api/plagiarism_api.py).");
    } finally {
      setIsChecking(false);
    }
  };

  const handleCheckAI = async () => {
    if (!text.trim()) {
      alert("Please enter some text to check for AI content");
      return;
    }

    setIsChecking(true);

    try {
      const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5001';
      
      const response = await fetch(`${API_BASE}/api/ai-check`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const result = await response.json();

      // Transform result to match ResultsPage expectations
      const aiResult = {
        type: 'ai_check',
        combined_prob_ai: result.data?.combined_prob_ai || 0,
        combined_label: result.data?.combined_label || 'Thấp -Người Viết',
        sentence_analysis: result.data?.sentence_analysis || [],
        high_risk_sentences: result.data?.high_risk_sentences || [],
        stats: result.data?.stats || {},
        analyzedText: text,
        wordCount: text.trim().split(/\s+/).length,
      };

      // Attempt to save history + report if authenticated
      try {
        if (token) {
          const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5001';
          await fetch(`${API_BASE_URL}/api/history`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${token}`,
            },
            body: JSON.stringify({ text, result: aiResult, inputType: 'text' }),
          });
        }
      } catch (e) {
        console.warn('Failed to save history/report:', e);
      }

      navigate("/results", { state: { result: aiResult } });
    } catch (error) {
      console.error("Error checking AI content:", error);
      alert("An error occurred while checking for AI content. Please make sure the Node.js server is running.");
    } finally {
      setIsChecking(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-gray-50">
      {/* Navigation */}
      <Header />

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-4 py-8">
        <div className="mb-6">
          <Link to="/">
            <Button variant="ghost" size="sm">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Home
            </Button>
          </Link>
        </div>

        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Plagiarism Checker</h1>
          <p className="text-muted-foreground">
            Paste your text, upload a document, or try a sample to check for plagiarism
          </p>
        </div>

        <Card className="p-6">
          <Tabs defaultValue="text">
            <TabsList className="grid w-full grid-cols-3 mb-4">
              <TabsTrigger triggerValue="text">
                <FileText className="w-4 h-4 mr-2" />
                Text
              </TabsTrigger>
              <TabsTrigger triggerValue="file">
                <Upload className="w-4 h-4 mr-2" />
                Upload
              </TabsTrigger>
              <TabsTrigger triggerValue="samples">
                <Sparkles className="w-4 h-4 mr-2" />
                Samples
              </TabsTrigger>
            </TabsList>

            <TabsContent contentValue="text">
              <Textarea
                placeholder="Paste your text here..."
                value={text}
                onChange={handleTextChange}
                className="min-h-[400px] mb-4"
              />
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">
                  Word count: {wordCount}
                </span>
                <div className="flex gap-2">
                  <Button
                    onClick={handleCheckAI}
                    disabled={isChecking || !text.trim()}
                    size="lg"
                    variant="outline"
                  >
                    {isChecking ? "Checking..." : "Check for AI"}
                  </Button>
                  <Button
                    onClick={handleCheck}
                    disabled={isChecking || !text.trim()}
                    size="lg"
                  >
                    {isChecking ? "Checking..." : "Check for Plagiarism"}
                  </Button>
                </div>
              </div>
            </TabsContent>

            <TabsContent contentValue="file">
              <div 
                className={`border-2 border-dashed rounded-lg p-12 text-center mb-4 transition-colors ${
                  isDragging ? 'border-primary bg-primary/5' : 'border-gray-300'
                }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                <p className="mb-4 text-muted-foreground">
                  {isDragging ? 'Drop your file here' : 'Drag and drop your file here, or click to browse'}
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  onChange={handleFileChange}
                  accept=".txt,.pdf,.doc,.docx"
                  id="file-upload-input"
                  style={{ display: 'none' }}
                />
                <Button 
                  variant="outline" 
                  type="button"
                  onClick={(e) => {
                    e.preventDefault();
                    console.log("Button clicked!");
                    const input = document.getElementById('file-upload-input');
                    console.log("Input element:", input);
                    if (input) {
                      input.click();
                      console.log("Input clicked!");
                    }
                  }}
                >
                  Choose File
                </Button>
                {selectedFile && (
                  <p className="mt-4 text-sm text-muted-foreground">
                    Selected: {selectedFile.name}
                  </p>
                )}
              </div>
              {text && (
                <>
                  <Textarea
                    value={text}
                    onChange={handleTextChange}
                    className="min-h-[300px] mb-4"
                  />
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">
                      Word count: {wordCount}
                    </span>
                    <div className="flex gap-2">
                      <Button
                        onClick={handleCheckAI}
                        disabled={isChecking || !text.trim()}
                        size="lg"
                        variant="outline"
                      >
                        {isChecking ? "Checking..." : "Check for AI"}
                      </Button>
                      <Button
                        onClick={handleCheck}
                        disabled={isChecking || !text.trim()}
                        size="lg"
                      >
                        {isChecking ? "Checking..." : "Check for Plagiarism"}
                      </Button>
                    </div>
                  </div>
                </>
              )}
            </TabsContent>

            <TabsContent contentValue="samples">
              <div className="space-y-4 mb-4">
                {SAMPLE_TEXTS.map((sample, index) => (
                  <Card
                    key={index}
                    className="p-4 cursor-pointer hover:bg-accent transition-colors"
                    onClick={() => handleSampleClick(sample.text)}
                  >
                    <h3 className="font-semibold mb-2">{sample.title}</h3>
                    <p className="text-sm text-muted-foreground line-clamp-2">
                      {sample.text}
                    </p>
                  </Card>
                ))}
              </div>
              {text && (
                <>
                  <Textarea
                    value={text}
                    onChange={handleTextChange}
                    className="min-h-[300px] mb-4"
                  />
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">
                      Word count: {wordCount}
                    </span>
                    <div className="flex gap-2">
                      <Button
                        onClick={handleCheckAI}
                        disabled={isChecking || !text.trim()}
                        size="lg"
                        variant="outline"
                      >
                        {isChecking ? "Checking..." : "Check for AI"}
                      </Button>
                      <Button
                        onClick={handleCheck}
                        disabled={isChecking || !text.trim()}
                        size="lg"
                      >
                        {isChecking ? "Checking..." : "Check for Plagiarism"}
                      </Button>
                    </div>
                  </div>
                </>
              )}
            </TabsContent>
          </Tabs>
        </Card>
      </div>
    </div>
  );
};

export default CheckerPage;
