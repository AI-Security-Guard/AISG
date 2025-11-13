import styled from "styled-components";

export const DeveloperContainer = styled.div`
    min-height: 100vh;
    display: flex;
    align-items: center;
`;

export const SlideBox = styled.div`
    background-color: rgba(248, 250, 252, 0.9);
    width: 100%;
    max-width: 700px;
    margin: 0 auto;
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    justify-content: center;
    display: flex;
    align-items: center;
    flex-direction: column;
`;

export const Tag = styled.span`
    display: inline-block;
    background: #e2e8f0;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    margin-left: 6px;
    font-weight: bold;
`;
