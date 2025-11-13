import React, { useState } from "react";
import * as S from "./LoginPage.style";
import Header from "../../component/Header/Header.js";
import LongButton from "../../component/LongButton/LongButton.js";
import Input from "../../component/Input/Input.js";
import { useNavigate } from "react-router-dom";
import CustomModal from "../../component/CustomModal/CustomModal.js";
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";
import axios from "axios";

function LoginPage() {
    const navigate = useNavigate();
    const [id, setId] = useState("");
    const [password, setPassword] = useState("");
    const [modalOpen, setModalOpen] = useState(false);

    const handleLogin = async () => {
        if (!id || !password) {
            setModalOpen(true);
            return;
        }

        try {
            const response = await axios.post("http://127.0.0.1:5000/login", {
                username: id,
                password: password,
            });

            localStorage.setItem("user", JSON.stringify(response.data.user));
            localStorage.setItem("access_token", response.data.access_token);
            navigate("/render");
            console.log(response.data);
        } catch (error) {
            setModalOpen(true);
        }
    };

    return (
        <>
            <Header />
            <S.Container>
                <S.LoginBox>
                    <S.Title>로그인</S.Title>
                    <Input label="아이디" variant="outlined" value={id} onChange={(e) => setId(e.target.value)} />
                    <Input
                        label="비밀번호"
                        type="password"
                        variant="outlined"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                    />
                    <LongButton txt="로그인" onClick={handleLogin} />
                </S.LoginBox>
            </S.Container>
            <CustomModal
                open={modalOpen}
                onClose={() => setModalOpen(false)}
                title="로그인 실패"
                message="아이디/비밀번호를 확인해주세요."
                icon={<ErrorOutlineIcon style={{ fontSize: 60, color: "#6E6E6E" }} />}
                buttons={[{ label: "확인", onClick: () => setModalOpen(false) }]}
            />
        </>
    );
}

export default LoginPage;
