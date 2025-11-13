import React from "react";
import Header from "../../component/Header/Header.js";
import { DeveloperContainer, SlideBox, Tag } from "./DeveloperPage.style.js";
import { Swiper, SwiperSlide } from "swiper/react";
import "swiper/css";
import "swiper/css/pagination";
import { Pagination } from "swiper/modules";

function DeveloperPage() {
    const slideItems = [
        {
            title: "개발자",
            members: [
                { name: "변무혁", tags: ["AI"], intro: "DORO" },
                {
                    name: "박소정",
                    tags: ["Desgin", "Frontend", "Backend"],
                    intro: "안녕히 계세요~ 전 이 대학교의 모든 굴레와 속박을 벗어던지고 제 행복을 찾아 떠납니다.",
                },
                {
                    name: "남동현",
                    tags: ["Desgin", "Frontend"],
                    intro: "안녕하세요. AI 기반 거동수상자 감시 시스템 프로젝트에서 프론트엔드를 담당하고 있습니다.",
                },
                {
                    name: "전한나",
                    tags: ["AI", "배포"],
                    intro: "AI는 아직 배우는 중이지만, 저도 그래요. 같이 성장 중.",
                },
                { name: "김효지", tags: ["AI", "Backend"], intro: "여러 분야를 공부하고 있습니다 :)" },
            ],
        },
        {
            title: "프로젝트 설명",
            content:
                "AI 방범대 사이트는 영상 분석 기술을 기반으로,\nCCTV나 녹화 영상을 자동으로 처리하여 사람의 행동을 인식하고 \n그중 이상 징후나 위험 행동을 보이는 거동 수상자를 탐지하는 서비스입니다.\n\n영상 프레임 단위 분석을 통해 인물의 움직임과 자세를 추적하고,\nAI 모델이 비정상적인 행동(폭행, 추격, 쓰러짐 등)을 판별하며,\n감지된 구간에 대한 상세 정보를 시각화하여 관리자에게 제공합니다.\n\n",
        },
    ];

    return (
        <>
            <Header />
            <DeveloperContainer>
                <Swiper modules={[Pagination]} pagination={{ clickable: true }} spaceBetween={30} slidesPerView={1}>
                    {slideItems.map((item, index) => (
                        <SwiperSlide key={index}>
                            <SlideBox>
                                <h2>{item.title}</h2>
                                {"members" in item ? (
                                    <ul>
                                        {item.members.map((member, i) => (
                                            <li key={i} style={{ marginBottom: "1rem" }}>
                                                <strong>{member.name}</strong>{" "}
                                                {member.tags.map((tag, j) => (
                                                    <Tag key={j}>{tag}</Tag>
                                                ))}
                                                <p style={{ marginTop: "0.3rem" }}>{member.intro}</p>
                                            </li>
                                        ))}
                                    </ul>
                                ) : (
                                    <p style={{ whiteSpace: "pre-line" }}>{item.content}</p>
                                )}
                            </SlideBox>
                        </SwiperSlide>
                    ))}
                </Swiper>
            </DeveloperContainer>
        </>
    );
}

export default DeveloperPage;
