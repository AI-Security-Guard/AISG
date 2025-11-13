import * as D from "./Detail.style.js";
import Header from "../../component/Header/Header.js";
import ShortButton from "../../component/ShortButton/ShortButton.js";
import React, { useEffect, useState } from "react";
import { useNavigate, useLocation, useParams } from "react-router-dom";

const API_BASE = "http://127.0.0.1:5001";

function DetailPage() {
    const navigate = useNavigate();
    const { clipId } = useParams();
    const { state: entryFromState } = useLocation();
    const [entry, setEntry] = useState(entryFromState || null);

    const toWebPath = (p) => (typeof p === "string" ? p.replace(/\\/g, "/") : p);
    const joinURL = (base, path) => `${base.replace(/\/$/, "")}/${(path || "").replace(/^\//, "")}`;
    const clipRel = entry?.clipUrl ?? entry?.clipPath ?? entry?.clip_url ?? entry?.clip ?? null;
    const thumbRel = entry?.thumbPath ?? entry?.thumb_url ?? entry?.thumb ?? null;
    const videoPath = clipRel ? joinURL(API_BASE, toWebPath(clipRel)) : null;
    const posterPath = thumbRel ? joinURL(API_BASE, toWebPath(thumbRel)) : "/image/default.png";

    const occurredAt = entry?.date ?? "알 수 없음";
    const type = entry?.message ?? "알 수 없음";
    const coordinate = entry?.start_bbox ?? "-";
    const startBBox = entry?.start_bbox;

    let coordinateText = "-";
    if (startBBox && startBBox.length === 4) {
        const [x1, y1, x2, y2] = startBBox;
        coordinateText = `(${x1}, ${y1}) ~ (${x2}, ${y2})`;
    }

    const GoBackButtonClick = () => navigate(-1);
    console.log("Detail entry:", entry);
    console.log("videoPath:", videoPath);
    console.log("posterPath:", posterPath);

    useEffect(() => {
        if (entry) return;
        const jobId = localStorage.getItem("jobId");
        if (!jobId) return;
        (async () => {
            try {
                const res = await fetch(`${API_BASE}/jobs/${jobId}/clips`);
                if (!res.ok) return;
                const data = await res.json();
                const found = (data.clips || []).find((c) => String(c.clip_id) === String(clipId));
                if (!found) return;
                setEntry({
                    id: found.clip_id,
                    date: found.start_time ?? "",
                    message: found.class_name === "assault" ? "폭행" : found.class_name ?? "",
                    start_bbox: found.start_bbox || found.bbox || null,
                    clipUrl: found.clip_url ?? found.clip_path ?? found.clip ?? found.clipl ?? "",
                    thumbPath: found.thumb_url ?? found.thumb ?? "",
                });
            } catch (e) {
                console.error(e);
            }
        })();
    }, [entry, clipId]);

    return (
        <>
            <D.DetailContainer>
                <Header />
                <D.Container>
                    <D.Detail>
                        <D.TempVideo controls>
                            <source src={videoPath} type="video/mp4" />
                        </D.TempVideo>
                        <D.DetailContent>
                            <D.Date>발생시기: {occurredAt}</D.Date>
                            <D.Type>수상행동 유형: {type}</D.Type>
                            <D.Location>최초 발생 위치: {coordinateText}</D.Location>
                        </D.DetailContent>
                    </D.Detail>

                    <D.ButtonContainer>
                        <ShortButton txt="뒤로가기" onClick={GoBackButtonClick} />
                    </D.ButtonContainer>
                </D.Container>
            </D.DetailContainer>
        </>
    );
}

export default DetailPage;
