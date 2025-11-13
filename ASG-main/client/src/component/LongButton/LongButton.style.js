import { styled } from "@mui/material/styles";
import Button from "@mui/material/Button";

export const CustomButton = styled(Button)(() => ({
    backgroundColor: "#94a3b8", // 차분한 블루그레이 (배경과 어울림)
    color: "#ffffff",
    borderRadius: "999px", // pill-shaped
    width: "100%",
    height: "3.2em",
    fontSize: "1.1rem",
    fontWeight: "bold",
    letterSpacing: "0.05em",
    textTransform: "none", // MUI 기본 대문자 방지
    boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
    transition: "all 0.2s ease-in-out",
    "&:hover": {
        backgroundColor: "#cbd5e1",
        color: "#111827",
        boxShadow: "0 6px 16px rgba(0, 0, 0, 0.15)",
    },
    "&:active": {
        backgroundColor: "#64748b",
        color: "#ffffff",
    },
}));
