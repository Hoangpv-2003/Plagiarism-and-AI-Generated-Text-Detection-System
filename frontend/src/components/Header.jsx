import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { Button } from "./ui/button";
import { useAuth } from "../lib/AuthContext";

const Header = () => {
  const navigate = useNavigate();
  const { isAuthenticated, logout } = useAuth();

  const handleLogout = () => {
    logout();
    navigate("/login", { replace: true });
  };

  return (
    <nav className="nav-container">
      <div className="nav-content">
        <div className="nav-items">
          <div className="flex items-center gap-6">
            <Link to="/" className="nav-logo">
              Checker
            </Link>
            <div className="flex items-center gap-4">
              <Link to="/plagiarism" className="nav-link">
                Kiểm tra đạo văn
              </Link>
              <Link to="/ai" className="nav-link">
                Kiểm tra AI
              </Link>
            </div>
          </div>

          <div className="nav-links">
            <Link to="/history" className="nav-link">
              Lịch sử
            </Link>

            {isAuthenticated ? (
              <Button variant="outline" size="sm" onClick={handleLogout}>
                Đăng xuất
              </Button>
            ) : (
              <Link to="/login">
                <Button variant="outline" size="sm">
                  Đăng nhập
                </Button>
              </Link>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Header;
