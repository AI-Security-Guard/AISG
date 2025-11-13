import ShortButton from "../../component/ShortButton/ShortButton.js";
import { MainContainer, Logo, LogoImg, LogoTxt } from "./MainPage.style.js";
import { useNavigate } from "react-router-dom";
import Footer from "../../component/Footer/Footer.js";
function MainPage() {
    const navigate = useNavigate();

    const GoStart = () => {
        navigate("/login");
    };

    return (
        <>
            <MainContainer>
                <Logo>
                    <LogoImg src="/image/cctv.png" alt="로고 이미지" />
                    <LogoTxt>AI방범대</LogoTxt>
                    <ShortButton txt="시작하기" onClick={GoStart} />
                </Logo>
                <Footer />
            </MainContainer>
        </>
    );
}

export default MainPage;
