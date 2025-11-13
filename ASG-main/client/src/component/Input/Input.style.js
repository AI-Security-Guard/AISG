import { styled } from "@mui/material/styles";
import TextField from "@mui/material/TextField";

export const CustomInput = styled(TextField)(({ theme }) => ({
    width: "100%",
    "& .MuiOutlinedInput-root": {
        borderRadius: "12px",
        backgroundColor: "rgba(255, 255, 255, 0.6)", // 여기만 배경 적용
        backdropFilter: "blur(6px)",
        "& fieldset": {
            borderColor: "rgba(200, 200, 200, 0.4)",
        },
        "&:hover fieldset": {
            borderColor: "#94a3b8",
        },
        "&.Mui-focused fieldset": {
            borderColor: "#64748b",
            borderWidth: "1.5px",
        },
    },
    "& .MuiFormHelperText-root": {
        marginLeft: "4px",
        marginTop: "4px",
        fontSize: "0.85rem",
        color: theme.palette.error.main,
        backgroundColor: "transparent", // ✅ 배경 제거
        padding: 0,
    },
}));
