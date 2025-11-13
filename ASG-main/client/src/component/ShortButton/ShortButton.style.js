import { styled } from "@mui/material/styles";
import Button from "@mui/material/Button";

export const CustomButton = styled(Button)(() => ({
    backgroundColor: "#94a3b8",
    color: "#ffffff",
    borderRadius: "10px",
    width: "17%",
    maxWidth: "200px",
    minWidth: "150px",
    height: "3.2em",
    fontSize: "1.1rem",
    fontWeight: "bold",
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
