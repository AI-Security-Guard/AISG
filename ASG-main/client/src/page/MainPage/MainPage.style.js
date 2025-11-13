import styled, { keyframes } from "styled-components";

const roll = keyframes`
  0% {
    transform: translateX(-200%) rotate(0deg);
  }
  100% {
    transform: translateX(0) rotate(360deg);
  }
`;

const fadeIn = keyframes`
  0% {
    opacity: 0;
  }
  100% {
    opacity: 1;
  }
`;

const float = keyframes`
  0%, 100% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-10px);
  }
`;

const gradientShift = keyframes`
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
`;

export const MainContainer = styled.div`
    flex-direction: column;
    height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;

    // background: linear-gradient(135deg, #e5e7eb, #cbd5e1, #94a3b8, #cbd5e1);
    // background-size: 300% 300%;
    // animation: ${gradientShift} 20s ease-in-out infinite;
`;

export const Logo = styled.div`
    flex-direction: column;
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 5%;
`;

export const LogoImg = styled.img`
    width: 20em;
    animation: ${roll} 1s ease-out, ${float} 3s ease-in-out infinite;
    animation-delay: 0s, 1s;
    animation-fill-mode: forwards;
`;

export const LogoTxt = styled.div`
    font-size: 3rem;
    font-weight: bold;
    opacity: 0;
    animation: ${fadeIn} 1s ease-out forwards;
    animation-delay: 1s;
`;
