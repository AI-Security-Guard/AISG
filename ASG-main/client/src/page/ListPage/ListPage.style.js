// ListPage.style.js
import styled, { keyframes } from "styled-components";
import { Checkbox, Pagination } from "@mui/material";

const gradientShift = keyframes`
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
`;

export const ListContainer = styled.div`
    width: 100%;
    height: 100vh;
    background: linear-gradient(135deg, #e5e7eb, #cbd5e1, #94a3b8, #cbd5e1);
    background-size: 300% 300%;
    background-position: 0% 50%;
    animation: ${gradientShift} 20s ease-in-out infinite;
`;

export const Container = styled.div`
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-around;
    position: absolute;
    bottom: 0;
    right: 0;
    width: 85%;
    height: 90%;
    padding: 1rem;
    box-sizing: border-box;
`;

export const UnconfirmedCountMessage = styled.div`
    font-size: 1.5rem;
    font-weight: bold;
    color: #1e293b;
    margin-bottom: 1rem;
    text-align: center;
    background: rgba(255, 255, 255, 0.6);
    padding: 0.8rem 1.2rem;
    border-radius: 12px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
`;

export const SuspectContainer = styled.div`
    width: 100%;
    height: 65%;
    display: flex;
    flex-direction: column;
    gap: 5%;
    align-items: center;
    justify-content: center;
`;

export const SuspectColumn = styled.div`
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    width: 90%;
    justify-items: center;
    gap: 1%;
`;

export const SuspectEntry = styled.div`
    display: flex;
    flex-direction: column;
    width: 90%;
    padding: 0.75rem;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
    transition: transform 0.2s ease, box-shadow 0.2s ease;

    &:hover {
        transform: scale(1.03) translateY(-2px);
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.12);
    }
`;

export const SuspectPoto = styled.img`
    width: 100%;
    border-radius: 10px 10px 0 0;
    border: 1px solid #ccc;
    height: 70%;
`;

export const SuspectDetail = styled.div`
    width: 100%;
    border-radius: 0 0 10px 10px;
    background: #f1f5f9;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid #ccc;
    height: 80%;
`;

export const IncidentInfo = styled.div`
    font-weight: 600;
    width: 100%;
    margin-left: 5%;
`;

export const Date = styled.div`
    color: #1e293b;
`;

export const DetailMessage = styled.div`
    font-size: 13px;
    color: #1e293b;
`;

export const ConfirmCheckbox = styled(Checkbox)``;

export const ListPagination = styled(Pagination)`
    & .MuiPaginationItem-page.Mui-selected {
        background-color: rgba(39, 39, 39, 0.5);
        color: black;
    }

    & .MuiPaginationItem-page:hover {
        background-color: rgba(39, 39, 39, 0.2);
        color: black;
    }

    & .MuiPaginationItem-page:active {
        background-color: rgba(39, 39, 39, 0.5);
        color: black;
    }
`;
