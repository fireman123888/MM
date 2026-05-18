"""Small public-domain style sample corpus for local MVP testing.

The snippets below are short excerpts/paraphrased legal provisions for
development and testing. Replace them with authorized, versioned legal sources
before production use.
"""

from __future__ import annotations

from .models import LegalDocument


def sample_documents() -> list[LegalDocument]:
    labor_contract_law = LegalDocument(
        doc_id="cn_labor_contract_law_sample",
        title="中华人民共和国劳动合同法（样例节选）",
        source_url="https://flk.npc.gov.cn/",
        metadata={"issuer": "全国人大常委会", "legal_rank": "法律", "tenant_id": "public"},
        text="""
第一章 总则

第十条 建立劳动关系，应当订立书面劳动合同。已建立劳动关系，未同时订立书面劳动合同的，应当自用工之日起一个月内订立书面劳动合同。

第四十七条 经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资的标准向劳动者支付。六个月以上不满一年的，按一年计算；不满六个月的，向劳动者支付半个月工资的经济补偿。

第四十八条 用人单位违反本法规定解除或者终止劳动合同，劳动者要求继续履行劳动合同的，用人单位应当继续履行；劳动者不要求继续履行或者劳动合同已经不能继续履行的，用人单位应当依照本法规定支付赔偿金。

第八十二条 用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的，应当向劳动者每月支付二倍的工资。
""",
    )

    arbitration_law = LegalDocument(
        doc_id="cn_labor_dispute_arbitration_law_sample",
        title="中华人民共和国劳动争议调解仲裁法（样例节选）",
        source_url="https://flk.npc.gov.cn/",
        metadata={"issuer": "全国人大常委会", "legal_rank": "法律", "tenant_id": "public"},
        text="""
第三章 仲裁

第二十七条 劳动争议申请仲裁的时效期间为一年。仲裁时效期间从当事人知道或者应当知道其权利被侵害之日起计算。
""",
    )

    social_insurance_law = LegalDocument(
        doc_id="cn_social_insurance_law_sample",
        title="中华人民共和国社会保险法（样例节选）",
        source_url="https://flk.npc.gov.cn/",
        metadata={"issuer": "全国人大常委会", "legal_rank": "法律", "tenant_id": "public"},
        text="""
第一章 总则

第五十八条 用人单位应当自用工之日起三十日内为其职工向社会保险经办机构申请办理社会保险登记。
""",
    )
    return [labor_contract_law, arbitration_law, social_insurance_law]
