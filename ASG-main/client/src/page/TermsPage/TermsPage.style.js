import styled from "styled-components";
import { Checkbox, FormControlLabel, TextField } from "@mui/material";

export const StyledFormControlLabel = styled(FormControlLabel)`
    margin-left: 10px;

    .MuiTypography-root {
        font-weight: bold;
        font-size: 1rem;
        color: #1e293b; /* 원하는 텍스트 색상 */
    }

    .MuiCheckbox-root {
        color: #1e293b;

        &:hover {
            background-color: rgba(30, 41, 59, 0.04);
        }
    }
`;

export const Container = styled.div`
    display: flex;
    justify-content: center;
    width: 100%;
    height: 100vh;
    align-items: center;
`;

export const Box = styled.div`
    width: 100%;
    max-width: 400px;
    min-width: 300px;
    background: rgba(255, 255, 255, 0.75);
    padding: 3%;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(45, 41, 41, 0.3);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
`;

export const Title = styled.h2`
    text-align: center;
`;

export const CheckLine = styled(Checkbox)``;

export const Section = styled.div`
    // background-color: #cbcccd;
    // padding: 20px;
    // margin-bottom: 20px;
    // border-radius: 5px;
`;

export const TextArea = styled(TextField).attrs({
    multiline: true,
    rows: 5,
})`
    width: 95%;
    overflow-y: auto;

    // 내부 textarea에 직접 스타일 지정 필요할 수도 있음
    & .MuiInputBase-root {
        padding: 10px;
        font-size: 14px;
        line-height: 1.4;
    }

    & textarea {
        resize: none;
        overflow-y: auto;
    }
`;

export const ButtonGroup = styled.div`
    display: flex;
    width: 95%;
    justify-content: flex-end;
    margin-top: 5%;
`;
