import React from "react";
import { Modal, Typography } from "@mui/material";
import * as S from "./CustomModal.style";
import ShortButton from "../../component/ShortButton/ShortButton.js";
import CloseIcon from "@mui/icons-material/Close";

function CustomModal({ open, onClose, title, message, icon, buttons = [] }) {
    return (
        <Modal open={open} onClose={onClose}>
            <S.ModalBox>
                <S.CloseButton onClick={onClose}>
                    <CloseIcon fontSize="inherit" />
                </S.CloseButton>
                <Typography variant="h6" component="h2" gutterBottom>
                    {title || "알림"}
                </Typography>
                {icon && <div style={{ margin: "8px 0" }}>{icon}</div>}
                <Typography sx={{ mb: 3 }}>{message}</Typography>
                <S.ButtonGroup>
                    {buttons.map((btn, idx) => (
                        <ShortButton key={idx} onClick={btn.onClick} txt={btn.label} />
                    ))}
                </S.ButtonGroup>
            </S.ModalBox>
        </Modal>
    );
}

export default CustomModal;
