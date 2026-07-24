# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> Với prompt "kể một sự thật thú vị về Hà Nội", ở nhiệt độ mặc định (0.7), Gemini 3.5 Flash trả lời rất mạch lạc và bám sát chủ đề (ví dụ đã ra được nội dung về kiến trúc "nhà ống" ở Phố Cổ). Theo quy luật chung của tham số này: temperature càng thấp (0.0), model càng chọn token có xác suất cao nhất ở mỗi bước, nên câu trả lời ổn định, gần như lặp lại giống nhau qua nhiều lần gọi; temperature càng cao (1.2–1.8), model bắt đầu chọn các token ít khả năng hơn, phản hồi đa dạng hơn nhưng dễ lạc chủ đề hoặc mất mạch lạc — thực tế mình quan sát được lúc đầu (trước khi sửa reasoning_effort) là ở 1.8, câu trả lời gần như chuyển sang nói lệch hẳn sang chủ đề khác. Ngưỡng "bắt đầu kém mạch lạc" với hầu hết model thường rơi vào khoảng 1.2–1.5 trở lên.
### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> Trợ lý soạn thảo hợp đồng pháp lý: temperature thấp (0.0–0.2) — pháp lý cần độ chính xác và nhất quán tuyệt đối, không chấp nhận "sáng tạo" vì có thể tạo ra điều khoản sai lệch nguy hiểm. Trợ lý viết slogan quảng cáo: temperature cao hơn (0.8–1.2) — mục tiêu là đa dạng ý tưởng, sự bất ngờ và mới lạ có giá trị hơn tính nhất quán.
### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> Workload: 20.000 người × 2 lần/ngày × 500 token output = 20.000.000 token/ngày. Model lớn (gpt-4o, $0.010/1K output): 20.000 × $0.010 = $200/ngày. Model nhỏ (gpt-4o-mini, $0.0006/1K output): 20.000 × $0.0006 = $12/ngày. Chênh lệch ~16.7 lần. Model lớn xứng đáng khi task cần suy luận phức tạp, độ chính xác cao (tư vấn pháp lý/y tế, phân tích nhiều bước). Model nhỏ là lựa chọn đúng cho task đơn giản, khối lượng lớn (phân loại, FAQ, tóm tắt ngắn).

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> Từ dữ liệu thật đã chạy được: persona "nhà thơ" mở đầu bằng hình ảnh ẩn dụ ("Hãy tưởng tượng một đứa trẻ ngồi bên hiên nhà..."), còn persona "kỹ sư senior" mở đầu trực tiếp, chuyên nghiệp ("Chào bạn, với tư cách là một kỹ sư..."). Về tổng thể, hai phản hồi khác nhau ở: giọng văn (ẩn dụ/cảm xúc vs. trực tiếp/chuyên môn), mức độ kỹ thuật (tránh thuật ngữ vs. dùng đúng thuật ngữ, có thể kèm ví dụ code), và cấu trúc mở đầu (kể chuyện vs. xưng danh vai trò). Từ đó rút ra: system prompt điều khiển được giọng văn, mức độ kỹ thuật, và cách trình bày — nhưng không đổi được tính đúng-sai của nội dung cốt lõi (cả hai đều phải giải thích đúng khái niệm machine learning).
### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> Số liệu thật đã chạy: đoạn văn 177 từ tiếng Việt. Ước lượng thô (177/0.75 ≈ 236 token) so với số token thật đếm bằng tiktoken (196 token) — chênh lệch −20.4%, nghĩa là ước lượng thô THỪA so với thực tế (không phải thiếu). Nếu dùng công thức số từ / 0.75 để dự toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán dư ra, có thể khiến ngân sách được duyệt cao hơn mức cần thiết — không nguy hiểm bằng việc dự toán thiếu, nhưng vẫn là sai số đáng kể (~20%) nên tốt nhất luôn dùng count_tokens thật thay vì công thức ước lượng khi lập ngân sách chính thức.
---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> Chatbot văn bản hưởng lợi nhiều nhất — người dùng thấy chữ xuất hiện ngay lập tức, giảm cảm giác chờ đợi dù tổng thời gian xử lý không đổi. Trợ lý giọng nói chỉ hưởng lợi nếu pipeline text-to-speech đọc theo từng câu ngay khi có (streaming TTS); nếu phải đợi toàn bộ văn bản rồi mới đọc thì streaming văn bản không tạo giá trị UX. Pipeline dịch tài liệu chạy ngầm ban đêm không cần streaming — không có ai đang chờ xem real-time, chỉ cần kết quả cuối cùng đúng và đầy đủ; thêm streaming chỉ tăng độ phức tạp code mà không có lợi ích tương ứng.
### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> Với delay cố định, hàng nghìn client sẽ retry đồng loạt tại cùng một mốc thời gian, tạo ra "đợt sóng" request tiếp theo đánh úp server đang quá tải — vấn đề lặp lại liên tục (hiện tượng "thundering herd", chính là thứ mình vừa gặp thực tế khi bị rate-limit 429 lúc chạy run_experiments.py). Exponential backoff giãn cách các lần retry ra xa dần, giảm áp lực tức thời, cho hệ thống thời gian phục hồi. Vấn đề còn sót lại: nhiều client vẫn có thể retry đồng bộ ngẫu nhiên cùng lúc vì họ bắt đầu request gần như cùng thời điểm ban đầu — kỹ thuật jitter (thêm độ trễ ngẫu nhiên vào mỗi lần backoff) phá vỡ sự đồng bộ này, trải đều các lần retry ra theo thời gian thay vì dồn cục tại các mốc cố định (0.1s, 0.2s, 0.4s...).
---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> Persona "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt" — xóa "trả lời ngắn gọn" khiến trợ lý trả lời dài dòng, giải thích thừa; xóa "bằng tiếng Việt" khiến trợ lý dễ chuyển sang trả lời tiếng Anh hoặc trộn ngôn ngữ không nhất quán.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> Nếu người dùng nói "Tôi đang học Python" ở lượt 1 rồi hỏi các câu khác trong 5 lượt tiếp theo, đến lượt 6 hỏi "gợi ý bài tập phù hợp trình độ của tôi" thì history[-8:] đã cắt mất lượt 1 nên trợ lý mất ngữ cảnh — khắc phục bằng cách duy trì một bản tóm tắt ngắn (running summary) các thông tin quan trọng, cập nhật định kỳ và luôn đưa vào system prompt thay vì chỉ dựa vào 8 message gần nhất.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
