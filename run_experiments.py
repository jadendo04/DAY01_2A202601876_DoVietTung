"""
Script chạy thí nghiệm cho exercises.md — Câu 1.1, 2.1, 2.2.
Chạy trên máy của bạn (nơi đã có .env với API key thật):

    python run_experiments.py

Yêu cầu: template.py và .env phải nằm cùng thư mục hoặc trong PYTHONPATH.
"""

import time

from template import call_openai, chat_with_system_prompt, count_tokens

SEPARATOR = "=" * 70
DELAY_BETWEEN_CALLS = 8  # giây — tránh rate limit của free tier


def experiment_1_1():
    """Câu 1.1 — Độ nhạy của temperature."""
    print(SEPARATOR)
    print("CÂU 1.1 — Độ nhạy của temperature")
    print(SEPARATOR)
    prompt = "Hãy kể cho tôi một sự thật thú vị về Hà Nội."
    for temp in [0.0, 0.7, 1.2, 1.8]:
        # max_tokens tăng lên vì Gemini 3.5 tính cả "thinking token" vào
        # max_tokens — nếu để 256 như mặc định, thinking ăn hết ngân sách
        # và câu trả lời hiển thị bị cắt cụt.
        text, latency = call_openai(prompt, temperature=temp, max_tokens=1024)
        print(f"\n--- temperature={temp} (latency={latency:.2f}s) ---")
        print(text)
        time.sleep(DELAY_BETWEEN_CALLS)
    print()


def experiment_2_1():
    """Câu 2.1 — Sức mạnh của persona."""
    print(SEPARATOR)
    print("CÂU 2.1 — Sức mạnh của persona")
    print(SEPARATOR)
    question = "Giải thích máy học (machine learning) là gì?"

    persona_1 = (
        "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, "
        "tránh thuật ngữ."
    )
    persona_2 = (
        "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code "
        "khi phù hợp."
    )

    text_1, latency_1 = chat_with_system_prompt(persona_1, question)
    print(f"\n--- Persona: Nhà thơ (latency={latency_1:.2f}s) ---")
    print(text_1)

    time.sleep(DELAY_BETWEEN_CALLS)

    text_2, latency_2 = chat_with_system_prompt(persona_2, question)
    print(f"\n--- Persona: Kỹ sư senior (latency={latency_2:.2f}s) ---")
    print(text_2)
    print()


def experiment_2_2():
    """Câu 2.2 — tiktoken vs đếm từ."""
    print(SEPARATOR)
    print("CÂU 2.2 — tiktoken vs đếm từ")
    print(SEPARATOR)

    # Thay đoạn văn ~150 từ tiếng Việt của bạn vào đây nếu muốn.
    sample_text = (
        "Trí tuệ nhân tạo đang thay đổi cách con người làm việc và học tập "
        "trong mọi lĩnh vực của đời sống. Từ y tế, giáo dục cho đến tài "
        "chính, các mô hình ngôn ngữ lớn giúp tự động hóa nhiều tác vụ vốn "
        "tốn nhiều thời gian và công sức của con người. Tuy nhiên, việc áp "
        "dụng công nghệ này cũng đặt ra không ít thách thức, bao gồm vấn đề "
        "về chi phí vận hành, độ chính xác của thông tin, và khả năng bảo "
        "mật dữ liệu người dùng. Nhiều doanh nghiệp Việt Nam đã bắt đầu thử "
        "nghiệm tích hợp các trợ lý ảo dựa trên AI vào quy trình chăm sóc "
        "khách hàng, nhằm giảm tải cho đội ngũ nhân viên và tăng tốc độ "
        "phản hồi. Dù vậy, để triển khai hiệu quả, các tổ chức cần đầu tư "
        "nghiêm túc vào việc huấn luyện dữ liệu phù hợp với ngữ cảnh và văn "
        "hóa địa phương, thay vì chỉ sao chép mô hình có sẵn từ nước ngoài."
    )

    word_count = len(sample_text.split())
    estimated_tokens = word_count / 0.75
    actual_tokens = count_tokens(sample_text)

    diff_pct = (actual_tokens - estimated_tokens) / actual_tokens * 100

    print(f"\nSố từ: {word_count}")
    print(f"Ước lượng thô (số từ / 0.75): {estimated_tokens:.1f} token")
    print(f"Số token thật (tiktoken):      {actual_tokens} token")
    print(f"Chênh lệch: {diff_pct:.1f}% "
          f"({'ước lượng thô THIẾU' if diff_pct > 0 else 'ước lượng thô THỪA'})")
    print()


if __name__ == "__main__":
    experiment_1_1()
    time.sleep(DELAY_BETWEEN_CALLS)
    experiment_2_1()
    experiment_2_2()
