import styled from 'styled-components';

export const HeaderContainer = styled.div`
    position: fixed;
    top: 0;
    width: 100%;
    height: 110px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    z-index: 10;

    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(255, 255, 255, 0.5);
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
`;

export const Logo = styled.div`
    display: flex;
    align-items: center;
    margin-left: 1.2rem;
    cursor: pointer;
`;

export const LogoImg = styled.img`
    width: 70px;
`;

export const LogoTxt = styled.div`
    font-size: 1.2rem;
    font-weight: bold;
`;

export const UserAuthBox = styled.div`
    display: flex;
    gap: 10%;
    align-items: center;
    width: 20em;
    justify-content: flex-end;
    margin-right: 1.2rem;
`;

export const LoginButton = styled.div`
    cursor: pointer;
`;

export const RegisterButton = styled.div`
    cursor: pointer;
`;

export const WelcomeMessage = styled.div``;

export const Logout = styled.div`
    cursor: pointer;
`;
