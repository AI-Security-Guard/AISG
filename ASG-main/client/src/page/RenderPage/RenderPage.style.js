import styled from "styled-components";

export const MainLayout = styled.div`
    display: flex;
    height: 100vh;
    width: 100%;
`;

export const ContentArea = styled.div`
    position: absolute;
    bottom: 0;
    right: 0;
    width: 90%;
    height: 90%;
    display: flex;
    justify-content: space-around;
    align-items: center;
    flex-direction: column;
`;

export const PlusIcon = styled.img`
    cursor: pointer;
`;

export const VideoPlayer = styled.video`
    width: 60%;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
`;

export const ButtonWrapper = styled.div`
    display: flex;
    justify-content: center;
    width: 80%;
`;

export const DeleteWrapper = styled.div`
    display: flex;
    justify-content: flex-end;
    width: 100%;
`;

export const DeleteVideo = styled.img`
    cursor: pointer;
    margin-right: 2%;
    margin-bottom: 1%;
    transition: transform 0.2s ease-in-out;

    &:hover {
        transform: scale(1.1);
    }
`;
