import React, { useEffect, useState } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import { Button } from "../components/ui/button";
import { Card } from "../components/ui/card";
import { ArrowLeft, AlertCircle } from "lucide-react";
import HighlightedText from "../components/HighlightedText";
import Header from "../components/Header";
import { useAuth } from "../lib/AuthContext";

const ReportPage = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [history, setHistory] = useState(null);
  const [report, setReport] = useState(null);
  const { token, isAuthenticated } = useAuth();

  const API_BASE = process.env.REACT_APP_API_BASE_URL || "http://localhost:5001";

  useEffect(() => {
    if (!token) {
      setError("Vui lòng đăng nhập để xem báo cáo.");
      setLoading(false);
      return;
    }
    (async () => {
      try {
        setLoading(true);
        const res = await fetch(`${API_BASE}/api/history/${id}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data?.message || "Tải báo cáo thất bại");
        setHistory(data.history);
        setReport(data.report);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    })();
  }, [id, token]);

  const handleLogout = () => {
    try {
      localStorage.removeItem("token");
      localStorage.removeItem("user");
    } catch (_) {}
    navigate("/login", { replace: true });
  };

  // Normalize from stored report to display like ResultsPage
  const normalized = report
    ? {
        is_plagiarism: report.isPlagiarism,
        confidence: report.confidence,
        threshold: report.threshold,
        best_match: report.bestMatch,
        top_matches: report.topMatches || [],
        sentence_analysis: report.sentenceAnalysis || [],
        stats: report.stats || {},
      }
    : null;

  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-gray-50">
      <Header />

      <div className="max-w-6xl mx-auto px-4 py-8">
        <div className="mb-6">
          <Link to="/history">
            <Button variant="ghost" size="sm">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Quay lại Lịch sử
            </Button>
          </Link>
        </div>

        {error && (
          <Card className="p-8 text-center">
            <AlertCircle className="w-12 h-12 mx-auto mb-3 text-red-500" />
            <div className="text-red-600 font-medium mb-2">{error}</div>
            <div className="text-sm text-muted-foreground">Hãy đảm bảo bạn đã đăng nhập.</div>
          </Card>
        )}

        {loading && <Card className="p-8">Đang tải báo cáo...</Card>}

        {!loading && !error && history && normalized && (
          <>
            <div className="mb-4">
              <h1 className="text-3xl font-bold">Chi tiết báo cáo</h1>
              <p className="text-muted-foreground mt-1">
                {history.title || "Chưa đặt tên"} • {history.wordCount} từ • {new Date(history.createdAt).toLocaleString()}
              </p>
            </div>

            {/* Summary Cards */}
            <div className="grid md:grid-cols-3 gap-6 mb-8">
              <Card className={`p-6 ${normalized.is_plagiarism ? 'bg-red-50 border-2 border-red-200' : 'bg-green-50 border-2 border-green-200'}`}>
                <div className="text-center">
                  <div className={`text-3xl font-bold ${normalized.is_plagiarism ? 'text-red-600' : 'text-green-600'} mb-2`}>
                    {normalized.is_plagiarism ? 'CÓ DẤU HIỆU ĐẠO VĂN' : 'KHÔNG ĐẠO VĂN'}
                  </div>
                  <div className="text-sm text-muted-foreground">Trạng thái</div>
                </div>
              </Card>

              <Card className="p-6">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-lg font-semibold">Điểm Confidence</h3>
                  <span className={`text-3xl font-bold ${normalized.confidence >= 0.6 ? 'text-red-600' : 'text-green-600'}`}>
                    {(normalized.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-3 w-full bg-gray-200 rounded overflow-hidden">
                  <div className={`h-full ${normalized.confidence >= 0.6 ? 'bg-red-500' : 'bg-green-500'}`} style={{ width: `${normalized.confidence * 100}%` }} />
                </div>
                <p className="text-sm text-muted-foreground mt-2">
                  Ngưỡng: {(normalized.threshold * 100).toFixed(0)}%
                </p>
              </Card>

              <Card className="p-6">
                <div className="text-sm text-muted-foreground">Điểm nguồn tương đồng tốt nhất</div>
                <div className="text-3xl font-bold text-orange-600">
                  {normalized.best_match?.final_score ? (normalized.best_match.final_score * 100).toFixed(1) : '0.0'}%
                </div>
                {normalized.best_match?.title && (
                  <div className="mt-2 text-sm">{normalized.best_match.title}</div>
                )}
              </Card>
            </div>

            {/* Top matches */}
            {normalized.top_matches && normalized.top_matches.length > 0 && (
              <Card className="p-6 mb-8">
                <h2 className="text-2xl font-bold mb-4">Top Nguồn Tương Đồng</h2>
                <div className="space-y-3">
                  {normalized.top_matches.map((m, i) => (
                    <div key={i} className="border-l-4 border-orange-500 pl-4 py-2 bg-orange-50 rounded">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-semibold">#{i + 1} - {m.title || m.doc_id}</span>
                        <span className="text-sm text-orange-600 font-medium">{(m.score * 100).toFixed(1)}%</span>
                      </div>
                      {m.url && (
                        <a href={m.url} target="_blank" rel="noopener noreferrer" className="text-xs text-blue-600 hover:underline">
                          {m.url}
                        </a>
                      )}
                    </div>
                  ))}
                </div>
              </Card>
            )}

            {/* Analyzed Text */}
            <Card className="p-6">
              <h2 className="text-2xl font-bold mb-4">Văn Bản Đã Phân Tích</h2>
              {normalized.sentence_analysis && normalized.sentence_analysis.length > 0 ? (
                <HighlightedText sentenceAnalysis={normalized.sentence_analysis} />
              ) : (
                <pre className="whitespace-pre-wrap text-sm text-muted-foreground">{history.text}</pre>
              )}
            </Card>
          </>
        )}
      </div>
    </div>
  );
};

export default ReportPage;
