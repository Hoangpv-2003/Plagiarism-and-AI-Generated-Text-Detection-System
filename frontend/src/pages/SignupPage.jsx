import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Card } from "../components/ui/card";
import { useAuth } from "../lib/AuthContext";

const SignupPage = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    confirmPassword: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const { login } = useAuth();

  const API_BASE = process.env.REACT_APP_API_BASE_URL || "http://localhost:5001";

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (formData.password !== formData.confirmPassword) {
      setError("Mật khẩu xác nhận không khớp!");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/auth/signup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: formData.name,
          email: formData.email,
          password: formData.password,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.message || "Signup failed");
      }
      // If API returns token/user on signup, log in immediately; otherwise go to login.
      if (data?.token) {
        login(data.token, data.user);
        navigate("/");
      } else {
        alert("Signup thành công! Hãy đăng nhập.");
        navigate("/login");
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-gray-50 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <Link to="/" className="text-3xl font-bold text-primary">
            Checker
          </Link>
          <h1 className="text-2xl font-bold mt-6 mb-2">Tạo tài khoản</h1>
          <p className="text-muted-foreground">
            Đăng ký để bắt đầu sử dụng
          </p>
        </div>

        <Card className="p-8">
          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded p-2">
                {error}
              </div>
            )}
            <div>
              <label className="block text-sm font-medium mb-2">
                Họ và tên
              </label>
              <Input
                type="text"
                name="name"
                placeholder="Nguyễn Văn A"
                value={formData.name}
                onChange={handleChange}
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Email
              </label>
              <Input
                type="email"
                name="email"
                placeholder="you@example.com"
                value={formData.email}
                onChange={handleChange}
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Mật khẩu
              </label>
              <Input
                type="password"
                name="password"
                placeholder="••••••••"
                value={formData.password}
                onChange={handleChange}
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Xác nhận mật khẩu
              </label>
              <Input
                type="password"
                name="confirmPassword"
                placeholder="••••••••"
                value={formData.confirmPassword}
                onChange={handleChange}
                required
              />
            </div>

            <div className="text-sm">
              <label className="flex items-start gap-2">
                <input type="checkbox" className="mt-1 rounded" required />
                <span className="text-muted-foreground">
                  Tôi đồng ý với{" "}
                  <Link to="#" className="text-primary hover:underline">
                    Điều khoản dịch vụ
                  </Link>{" "}
                  và{" "}
                  <Link to="#" className="text-primary hover:underline">
                    Chính sách bảo mật
                  </Link>
                </span>
              </label>
            </div>

            <Button type="submit" className="w-full" size="lg" disabled={loading}>
              {loading ? "Đang tạo..." : "Tạo tài khoản"}
            </Button>
          </form>

          <div className="mt-6 text-center text-sm">
            <span className="text-muted-foreground">Đã có tài khoản? </span>
            <Link to="/login" className="text-primary hover:underline font-medium">
              Đăng nhập
            </Link>
          </div>
        </Card>

        <div className="text-center mt-6">
          <Link to="/" className="text-sm text-muted-foreground hover:text-primary">
            ← Về trang chủ
          </Link>
        </div>
      </div>
    </div>
  );
};

export default SignupPage;
