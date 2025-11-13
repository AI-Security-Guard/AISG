import styled from "styled-components";

export const OpenSourceContainer = styled.div`
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 100vh;
    padding: 8% 10% 5%;
    background: radial-gradient(circle at top, #f3f7ff, #eaf1ff, #ffffff);
    backdrop-filter: blur(10px);
    background: transparent;
`;

export const Title = styled.div`
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #1976d2, #42a5f5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 24px;
    text-align: center;
    letter-spacing: 0.5px;
`;

export const OpenSourceContent = styled.div`
    width: 80%;
    max-width: 900px;
    border-radius: 20px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
    padding: 50px;
    margin-top: 70px;
    margin-bottom: 60px;
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
`;

export const Section = styled.div`
    display: flex;
    flex-direction: column;
    gap: 20px;
`;

export const SourceList = styled.div`
    display: flex;
    flex-direction: column;
    gap: 14px;
`;

export const SourceItem = styled.div`
    display: flex;
    align-items: flex-start;
    gap: 18px;
    padding: 20px 24px;
    border-radius: 14px;
    background: linear-gradient(145deg, #f9fbff, #eef4ff);
    border: 2px solid transparent;
    box-shadow: 0 3px 8px rgba(0, 0, 0, 0.05);
    transition: all 0.35s ease;

    &:hover {
        transform: translateY(-4px);
        background: linear-gradient(145deg, #ffffff, #f0f6ff);
        // border-image: linear-gradient(90deg, #1976d2, #42a5f5) 1;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
    }

    a {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1976d2;
        text-decoration: none;
    }

    a:hover {
        text-decoration: underline;
    }

    p {
        margin-top: 6px;
        color: #333;
        font-size: 0.95rem;
        line-height: 1.5;
    }
`;

export const IconWrapper = styled.div`
    font-size: 1.8rem;
    margin-top: 4px;
`;
