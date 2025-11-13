import styled from "styled-components";

export const DetailContainer = styled.div`
    display: flex;
    flex-direction: column;
    width: 100%;
    min-height: 100vh;
`;

export const Container = styled.div`
    display: flex;
    flex-direction: column;
    justify-content: center;
    width: 100%;
    align-items: center;
    margin-top: 10%;
    margin-bottom: 5%;
`;

export const Detail = styled.div`
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 12px;
    width: 70%;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2.5rem;
    gap: 2rem;
    box-shadow: 0px 8px 24px rgba(0, 0, 0, 0.15);
`;

export const TempVideo = styled.video`
    width: 90%;
    padding: 1rem;
    border-radius: 10px;
    background-color: #000;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
`;

export const DetailContent = styled.div`
    background: rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: 10px;
    width: 90%;
    font-weight: 500;
    font-size: 1.1rem;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    color: #1e293b;
`;
export const Date = styled.div``;

export const Type = styled.div``;

export const Location = styled.div``;

export const ButtonContainer = styled.div`
    display: flex;
    justify-content: flex-end;
    width: 75%;
    margin-top: 3%;
`;
