import React from "react";
import { Link } from "react-router-dom";
import { Button } from "../components/ui/button";
import { Card } from "../components/ui/card";
import { CheckCircle, Shield, Clock, BarChart3 } from "lucide-react";
import Header from "../components/Header";

const HomePage = () => {
  return (
    <div className="page-wrapper">
      {/* Navigation */}
      <Header />

      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <h1 className="hero-title">Kiểm tra nội dung nhanh và chính xác</h1>
          <p className="hero-subtitle">
            Chọn "Kiểm tra đạo văn" hoặc "Kiểm tra AI" trên thanh điều hướng để bắt đầu.
          </p>
        </div>

        <div className="review-text">
          <div className="review-content">
            <div className="flex items-center gap-1">
              <span className="review-rating">4.5/5</span>
              <span>•</span>
              <span>Dựa trên 12,623 đánh giá</span>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="stats-section">
        <div className="stats-grid">
          <div className="stats-item">
            <div className="stats-number">99.9%</div>
            <div className="stats-label">Tỷ lệ chính xác</div>
          </div>
          <div className="stats-item">
            <div className="stats-number">10M+</div>
            <div className="stats-label">Tài liệu đã kiểm tra</div>
          </div>
          <div className="stats-item">
            <div className="stats-number">50K+</div>
            <div className="stats-label">Người dùng hoạt động</div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="features-section">
        <div className="features-container">
          <h2 className="features-title">Vì sao chọn Checker?</h2>
          <div className="features-grid">
            <Card className="card-feature">
              <div className="feature-card">
                <div className="feature-icon-wrapper">
                  <CheckCircle className="feature-icon" />
                </div>
                <div>
                  <h3 className="feature-title">Phát hiện chính xác</h3>
                  <p className="feature-description">
                    Thuật toán giúp phát hiện mức độ tương đồng và các dấu hiệu cần chú ý.
                  </p>
                </div>
              </div>
            </Card>

            <Card className="card-feature">
              <div className="feature-card">
                <div className="feature-icon-wrapper">
                  <Shield className="feature-icon" />
                </div>
                <div>
                  <h3 className="feature-title">Bảo mật & riêng tư</h3>
                  <p className="feature-description">
                    Ưu tiên quyền riêng tư và an toàn dữ liệu trong quá trình xử lý.
                  </p>
                </div>
              </div>
            </Card>

            <Card className="card-feature">
              <div className="feature-card">
                <div className="feature-icon-wrapper">
                  <Clock className="feature-icon" />
                </div>
                <div>
                  <h3 className="feature-title">Kết quả nhanh</h3>
                  <p className="feature-description">
                    Trả kết quả trong thời gian ngắn để bạn kịp thời chỉnh sửa.
                  </p>
                </div>
              </div>
            </Card>

            <Card className="card-feature">
              <div className="feature-card">
                <div className="feature-icon-wrapper">
                  <BarChart3 className="feature-icon" />
                </div>
                <div>
                  <h3 className="feature-title">Báo cáo chi tiết</h3>
                  <p className="feature-description">
                    Hiển thị mức độ tương đồng, câu có rủi ro và thông tin đối chiếu.
                  </p>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-50 border-t py-8">
        <div className="max-w-6xl mx-auto px-4 text-center text-muted-foreground">
          <p>© 2025 Checker. Bảo lưu mọi quyền.</p>
        </div>
      </footer>
    </div>
  );
};

export default HomePage;
