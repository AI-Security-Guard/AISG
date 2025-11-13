import styled from "styled-components";

export const SidebarContainer = styled.div`
    width: 15%;
    height: 90%;
    position: fixed;
    left: 0;
    bottom: 0;
    display: flex;
    flex-direction: column;
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255, 255, 255, 0.5);
    box-shadow: 2px 0 10px rgba(0, 0, 0, 0.1);
    justify-content: space-between;
`;

export const SidebarButton = styled.div`
    padding: 10%;
    font-size: 1rem;
    font-weight: bold;
    color: black;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    transition: all 0.2s ease-in-out;

    &:hover {
        background-color: rgba(255, 255, 255, 0.5);
    }
    .content {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        transition: transform 0.2s ease-in-out;
    }

    &:hover .content {
        transform: scale(1.1);
    }
    &:active .content {
        transform: scale(0.95);
    }
`;

export const SidebarFooter = styled.div`
    font-size: 0.7rem;
    text-align: center;
    margin-bottom: 1rem;
`;
