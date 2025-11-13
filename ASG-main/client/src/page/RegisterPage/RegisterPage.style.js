import styled from "styled-components";

export const Container = styled.div`
    width: 100%;
    height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
`;

export const RegisterBox = styled.div`
    display: flex;
    flex-direction: column;
    justify-content: space-around;
    height: 45%;
    width: 100%;
    max-width: 400px;
    min-width: 300px;
    background: rgba(255, 255, 255, 0.75);
    padding: 3%;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.3);
`;

export const Title = styled.h2`
    font-size: 24px;
    text-align: center;
`;
