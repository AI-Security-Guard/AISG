import styled, { keyframes } from "styled-components";

const gradientShift = keyframes`
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
`;

export const BackgroundWrapper = styled.div`
    background: linear-gradient(135deg, #e5e7eb, #cbd5e1, #94a3b8, #cbd5e1);
    background-size: 300% 300%;
    background-position: 0% 50%;
    animation: ${gradientShift} 15s ease-in-out infinite;
`;
