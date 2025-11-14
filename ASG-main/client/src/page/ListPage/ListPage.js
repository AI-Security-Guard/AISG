import React, { useState, useEffect } from "react";
import {
    ListContainer,
    UnconfirmedCountMessage,
    SuspectEntry,
    SuspectPoto,
    SuspectDetail,
    Date,
    DetailMessage,
    ConfirmCheckbox,
    IncidentInfo,
    SuspectContainer,
    SuspectColumn,
    Container,
    ListPagination,
} from "./ListPage.style.js";
import Header from "../../component/Header/Header.js";
import Sidebar from "../../component/Sidebar/Sidebar.js";
import { useNavigate, useParams } from "react-router-dom";
function ListPage() {
    const { jobId } = useParams();
    const navigate = useNavigate();
    const [entries, setEntries] = useState([]);
    const [currentPage, setCurrentPage] = useState(1);
    const itemsPerPage = 8;
    const API_BASE = "http://127.0.0.1:5001";

    const fetchClipsByJobId = async (realJobId) => {
        const res = await fetch(`${API_BASE}/jobs/${realJobId}/clips`);
        if (!res.ok) {
            console.error("Failed to fetch clips", res.status);
            return;
        }
        const data = await res.json();
        const mapped = (data.clips || []).map((c, i) => ({
            id: c.id,
            date: c.start_time ?? "",
            message: c.class_name === "assault" ? "폭행" : c.class_name ?? "",
            checked: Boolean(c.checked),
            clipPath: (c.clip_path || "").replace(/\\/g, "/"),
            thumbPath: c.thumb_url || "",
            start_bbox: c.start_bbox || c.bbox || null,
            clipUrl: c.clip_url || "",
        }));
        setEntries(mapped);
    };

    useEffect(() => {
        (async () => {
            // 1) URL에 제대로 jobId가 온 경우
            if (jobId && jobId !== "null") {
                await fetchClipsByJobId(jobId);
                return;
            }

            // 2) URL이 /List/null 이거나 비어있을 때 → 서버한테 "이 사용자 최신 job" 물어보기
            const userStr = localStorage.getItem("user");
            if (!userStr) return;
            const user = JSON.parse(userStr);
            const username = user?.username;
            if (!username) return;

            const latestRes = await fetch(`${API_BASE}/jobs/latest?username=${encodeURIComponent(username)}`);
            if (!latestRes.ok) {
                console.error("no latest job for user");
                return;
            }
            const latestData = await latestRes.json();
            if (!latestData.job_id) return;

            await fetchClipsByJobId(latestData.job_id);
        })().catch((e) => console.error(e));
    }, [jobId]);

    const handlePageChange = (event, page) => {
        setCurrentPage(page);
    };

    const sortedEntries = [...entries].sort((a, b) => a.checked - b.checked);
    const indexOfLastEntry = currentPage * itemsPerPage;
    const indexOfFirstEntry = indexOfLastEntry - itemsPerPage;
    const currentEntries = sortedEntries.slice(indexOfFirstEntry, indexOfLastEntry);
    const unconfirmedCount = entries.filter((entry) => !entry.checked).length;

    const handleClick = async (entry) => {
        try {
            // 1) 서버에 체크 반영 (확정형)
            await fetch(`${API_BASE}/clips/${entry.id}/check`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
            });

            // 2) 프론트 상태 동기화
            setEntries((prev) => prev.map((e) => (e.id === entry.id ? { ...e, checked: true } : e)));

            // 3) 상세 이동(상태 최신 반영해서 넘기기)
            navigate(`/Detail/${entry.id}`, { state: { ...entry, checked: true } });
        } catch (e) {
            console.error("체크 업데이트 실패:", e);
            // 실패해도 상세로는 이동할 수 있게 하려면 아래 유지
            navigate(`/Detail/${entry.id}`, { state: entry });
        }
    };

    return (
        <>
            <ListContainer>
                <Header />
                <Sidebar />
                <Container>
                    <UnconfirmedCountMessage>
                        확인하지 않은 거동 수상자 목록이 {unconfirmedCount}개 있습니다.
                    </UnconfirmedCountMessage>

                    <SuspectContainer>
                        <SuspectColumn>
                            {currentEntries.slice(0, 4).map((entry) => (
                                <SuspectEntry key={entry.id} onClick={() => handleClick(entry)}>
                                    {/* 썸네일이 있으면 썸네일, 없으면 임시 이미지 */}
                                    <SuspectPoto
                                        as="img"
                                        src={entry.thumbPath ? `${API_BASE}${entry.thumbPath}` : "/image/poto.png"}
                                        alt="썸네일"
                                    />
                                    <SuspectDetail>
                                        <IncidentInfo>
                                            <Date>{entry.date}</Date>
                                            <DetailMessage>
                                                <span style={{ color: "red" }}>{entry.message}</span>이(가)
                                                발생했습니다.
                                            </DetailMessage>
                                        </IncidentInfo>
                                        <ConfirmCheckbox checked={entry.checked} disabled={true} />
                                    </SuspectDetail>
                                </SuspectEntry>
                            ))}
                        </SuspectColumn>

                        <SuspectColumn>
                            {currentEntries.slice(4, 8).map((entry) => (
                                <SuspectEntry key={entry.id} onClick={() => handleClick(entry)}>
                                    <SuspectPoto
                                        as="img"
                                        src={entry.thumbPath ? `${API_BASE}${entry.thumbPath}` : "/image/poto.png"}
                                        alt="썸네일"
                                    />
                                    <SuspectDetail>
                                        <IncidentInfo>
                                            <Date>{entry.date}</Date>
                                            <DetailMessage>
                                                <span style={{ color: "red" }}>{entry.message}</span>이(가)
                                                발생했습니다.
                                            </DetailMessage>
                                        </IncidentInfo>
                                        <ConfirmCheckbox checked={entry.checked} disabled={true} />
                                    </SuspectDetail>
                                </SuspectEntry>
                            ))}
                        </SuspectColumn>
                    </SuspectContainer>

                    <ListPagination
                        count={Math.ceil(sortedEntries.length / itemsPerPage)}
                        page={currentPage}
                        onChange={handlePageChange}
                    />
                </Container>
            </ListContainer>
        </>
    );
}

export default ListPage;
