import {
    FooterContainer,
    GitHubDiv,
    GitHubLogo,
    GitHubLetter,
    FooterInfo,
    ServiceStart,
    OpenSource,
    Developer,
} from "./Footer.style.js";
import { useNavigate } from "react-router-dom";

function Footer() {
    const navigate = useNavigate();

    const handleDeveloperClick = () => {
        navigate("/developer");
    };

    const handleOpenSourceClick = () => {
        navigate("/openSource");
    };

    return (
        <FooterContainer>
            <GitHubDiv>
                <GitHubLogo src="/image/github.png" alt="로고 이미지" />
                <GitHubLetter
                    as="a"
                    href="https://github.com/AI-Security-Guard/AI-Security-Guard"
                    target="_blank"
                    rel="noopener noreferrer"
                >
                    GitHub Repository
                </GitHubLetter>
            </GitHubDiv>
            <ServiceStart>ⓒ 2025 AI-Security-Guard All rights reserved</ServiceStart>
            <FooterInfo>
                <OpenSource onClick={handleOpenSourceClick}>오픈소스 라이선스</OpenSource>|
                <Developer onClick={handleDeveloperClick}>개발자 소개</Developer>
            </FooterInfo>
        </FooterContainer>
    );
}

export default Footer;
