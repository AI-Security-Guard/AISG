import { styled, keyframes } from "@mui/material/styles";
import { Box } from "@mui/material";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import { shouldForwardProp } from "@mui/system";

const spin = keyframes`
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
`;

// 로딩 애니메이션 (분석하기 버튼)
export const SpinnerWrapper = styled("div")({
    position: "relative",
    width: "60px",
    height: "60px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
});

export const Spinner = styled("div", {
    shouldForwardProp: (prop) => prop !== "visible",
})(({ visible }) => ({
    width: "40px",
    height: "40px",
    border: "6px solid rgba(0, 0, 0, 0.1)",
    borderTop: "6px solid #4a90e2",
    borderRadius: "50%",
    position: "absolute",
    top: 0,
    left: 0,

    animationName: spin,
    animationDuration: "1s",
    animationTimingFunction: "linear",
    animationIterationCount: "infinite",

    opacity: visible ? 1 : 0,
    transform: visible ? "scale(1)" : "scale(1.2)",
    transition: "opacity 0.5s ease, transform 0.5s ease",
}));

// 체크 아이콘 (분석하기 버튼)
export const CheckIcon = styled(CheckCircleIcon, {
    shouldForwardProp: (prop) => prop !== "visible",
})(({ visible }) => ({
    color: " #4a90e2",
    fontSize: "60px",
    position: "absolute",
    top: 0,
    left: 0,
    opacity: visible ? 1 : 0,
    transform: visible ? "scale(1)" : "scale(0.6)",
    transition: "opacity 0.5s ease, transform 0.5s ease",
}));

const fadeInModal = keyframes`
  from {
    opacity: 0;
    transform: translate(-50%, -50%) scale(0.95);
  }
  to {
    opacity: 1;
    transform: translate(-50%, -50%) scale(1);
  }
`;

export const ModalBox = styled(Box)(({ theme }) => ({
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    width: 420,
    minHeight: 280,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",

    background: "rgba(255, 255, 255, 0.5)",
    backdropFilter: "blur(18px)",
    WebkitBackdropFilter: "blur(18px)",
    border: "1px solid rgba(255, 255, 255, 0.3)",

    boxShadow: "0 16px 48px rgba(0, 0, 0, 0.15)",
    borderRadius: "20px",

    padding: "40px 30px",
    textAlign: "center",
    color: "#333333",
    fontFamily: "'Pretendard', sans-serif",

    animation: `${fadeInModal} 0.3s ease-out forwards`,

    "& h2": {
        fontSize: "1.6rem",
        fontWeight: 700,
        marginBottom: "1rem",
    },
    "& p": {
        fontSize: "1rem",
        color: "#444",
        marginBottom: "1.5rem",
    },
}));

export const ButtonGroup = styled("div")({
    display: "flex",
    gap: "12px",
    justifyContent: "center",
    flexWrap: "wrap",
});

export const CloseButton = styled("button")({
    position: "absolute",
    top: "16px",
    right: "16px",
    background: "transparent",
    border: "none",
    fontSize: "1.4rem",
    color: "#555",
    cursor: "pointer",
    transition: "color 0.2s ease",
    "&:hover": {
        color: "#000",
    },
});
