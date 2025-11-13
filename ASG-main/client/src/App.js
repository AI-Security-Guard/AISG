import "./App.css";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Background from "./component/Background/Background.js";
import MainPage from "./page/MainPage/MainPage.js";
import ListPage from "./page/ListPage/ListPage.js";
import DetailPage from "./page/DetailPage/DetailPage.js";
import TermsPage from "./page/TermsPage/TermsPage.js";
import RegisterPage from "./page/RegisterPage/RegisterPage.js";
import DeveloperPage from "./page/DeveloperPage/DeveloperPage.js";
import LoginPage from "./page/LoginPage/LoginPage.js";
import RenderPage from "./page/RenderPage/RenderPage";
import OpenSourcePage from "./page/OpenSourcePage/OpenSourcePage.js";

function App() {
    return (
        <>
            <Router>
                <Background>
                    <Routes>
                        <Route path="/login" element={<LoginPage />} />
                        <Route path="/" element={<MainPage />} />
                        <Route path="/List/:jobId" element={<ListPage />} />
                        <Route path="/Detail/:clipId" element={<DetailPage />} />
                        <Route path="/termspage" element={<TermsPage />} />
                        <Route path="/register" element={<RegisterPage />} />
                        <Route path="/render" element={<RenderPage />} />
                        <Route path="/Developer" element={<DeveloperPage />} />
                        <Route path="/openSource" element={<OpenSourcePage />} />
                    </Routes>
                </Background>
            </Router>
        </>
    );
}

export default App;
