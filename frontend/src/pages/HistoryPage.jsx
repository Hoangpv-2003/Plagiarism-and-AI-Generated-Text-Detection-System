import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Button } from "../components/ui/button";
import { Card } from "../components/ui/card";
import { FileText, Clock } from "lucide-react";
import Header from "../components/Header";
import { useAuth } from "../lib/AuthContext";

const HistoryPage = () => {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const { token, isAuthenticated } = useAuth();

  const API_BASE = process.env.REACT_APP_API_BASE_URL || "http://localhost:5001";

  useEffect(() => {
    if (!token) {
      setLoading(false);
      setItems([]);
      setError("Vui lòng đăng nhập để xem lịch sử.");
      return;
    }
    (async () => {
      try {
        setLoading(true);
        const res = await fetch(`${API_BASE}/api/history`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data?.message || "Tải lịch sử thất bại");
        setItems(Array.isArray(data.items) ? data.items : []);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    })();
  }, [token]);


  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-gray-50">
      {/* Navigation */}
      <Header />

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Lịch sử kiểm tra</h1>
          <p className="text-muted-foreground">
            Xem lại các lần kiểm tra trước đó
          </p>
        </div>

        {error && (
          <div className="mb-4 text-sm text-red-600 bg-red-50 border border-red-200 rounded p-3">
            {error}
          </div>
        )}

        {loading && (
          <Card className="p-6">Đang tải lịch sử...</Card>
        )}

        {!loading && items.length > 0 && (
          <div className="space-y-4">
            {items.map((item) => (
            <Card key={item.id} className="p-6 hover:shadow-lg transition-shadow">
              <div className="flex items-start justify-between">
                <div className="flex gap-4 flex-1">
                  <div className="flex items-center justify-center w-12 h-12 rounded-lg bg-primary/10">
                    <FileText className="w-6 h-6 text-primary" />
                  </div>
                  <div className="flex-1">
                    <h3 className="font-semibold text-lg mb-1">{item.title || "Chưa đặt tên"}</h3>
                    <div className="flex items-center gap-4 text-sm text-muted-foreground">
                      <span className="flex items-center gap-1">
                        <Clock className="w-4 h-4" />
                        {new Date(item.createdAt).toLocaleString()}
                      </span>
                      <span>{item.wordCount} từ</span>
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className={`text-2xl font-bold ${
                    item.similarity > 20 ? "text-red-600" :
                    item.similarity > 10 ? "text-yellow-600" :
                    "text-green-600"
                  }`}>
                    {item.similarity}%
                  </div>
                  <div className="text-sm text-muted-foreground">độ tương đồng</div>
                  <Link to={`/history/${item.id}`}>
                    <Button variant="outline" size="sm" className="mt-2">
                      Xem báo cáo
                    </Button>
                  </Link>
                </div>
              </div>
            </Card>
            ))}
          </div>
        )}

        {!loading && items.length === 0 && (
          <Card className="p-12 text-center">
            <FileText className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
            <h3 className="text-xl font-semibold mb-2">Chưa có lịch sử</h3>
            <p className="text-muted-foreground mb-6">
              Hãy bắt đầu kiểm tra để xem lịch sử tại đây
            </p>
            <Link to="/plagiarism">
              <Button>Kiểm tra tài liệu đầu tiên</Button>
            </Link>
          </Card>
        )}
      </div>
    </div>
  );
};

export default HistoryPage;
