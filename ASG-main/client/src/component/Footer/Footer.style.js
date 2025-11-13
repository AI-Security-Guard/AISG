import styled from "styled-components";

export const FooterContainer = styled.footer`
    display: flex;
    margin-top: auto;
    width: 100%;
    flex-direction: column;
    align-items: center;
    height: 15%;
    padding: 1%;
    gap: 1%;
    justify-content: space-around;
    background: rgba(255, 255, 255, 0.1);
    border-top: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 -1px 10px rgba(0, 0, 0, 0.1);
`;

export const GitHubDiv = styled.div`
    width: 100%;
    justify-content: center;
    display: flex;
    align-items: center;
    gap: 1%;
`;

export const GitHubLogo = styled.img``;

export const GitHubLetter = styled.div`
    font-size: 30px;
    font-weight: bold;
    color: black;
    text-decoration: none;
`;

export const FooterInfo = styled.div`
    display: flex;
    gap: 1%;
    width: 100%;
    justify-content: center;
`;

export const ServiceStart = styled.div``;

export const OpenSource = styled.div`
    cursor: pointer;
`;

export const Developer = styled.div`
    cursor: pointer;
`;
