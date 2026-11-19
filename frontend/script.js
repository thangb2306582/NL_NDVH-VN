const fileInput = document.getElementById("fileInput");
const uploadBox = document.getElementById("uploadBox");
const uploadContent = document.getElementById("uploadContent");
const preview = document.getElementById("preview");
const heatmapPreview = document.getElementById("heatmapPreview");
const predictBtn = document.getElementById("predictBtn");
const resetBtn = document.getElementById("resetBtn");

const resultText = document.getElementById("resultText");
const confidenceText = document.getElementById("confidenceText");
const confidenceBar = document.getElementById("confidenceBar");
const scoreList = document.getElementById("scoreList");

const festivalInfo = document.getElementById("festivalInfo");
const festivalTitle = document.getElementById("festivalTitle");
const festivalBadge = document.getElementById("festivalBadge");
const festivalDescription = document.getElementById("festivalDescription");
const festivalMeta = document.getElementById("festivalMeta");

const officialLinksContainer = document.getElementById(
  "officialLinksContainer",
);
const officialLinksList = document.getElementById("officialLinksList");

const heatmapToggleWrapper = document.getElementById("heatmapToggleWrapper");
const btnShowOriginal = document.getElementById("btnShowOriginal");
const btnShowHeatmap = document.getElementById("btnShowHeatmap");

// Khối so sánh 3 mô hình ở phía dưới ảnh
const modelsComparisonBox = document.getElementById("modelsComparisonBox");
const bestBadge = document.getElementById("bestBadge");
const modelsList = document.getElementById("modelsList");
const evaluationMetricNote = document.getElementById("evaluationMetricNote");

let selectedFile = null;
const apiUrl = "http://127.0.0.1:8000/predict";

// Mở hộp chọn file khi click vào ô upload
uploadBox.addEventListener("click", () => {
  if (!selectedFile) {
    fileInput.click();
  }
});

fileInput.addEventListener("change", (e) => {
  selectedFile = e.target.files[0];
  if (selectedFile) {
    preview.src = URL.createObjectURL(selectedFile);
    preview.classList.remove("d-none");
    heatmapPreview.classList.add("d-none");
    uploadContent.classList.add("d-none");
    heatmapToggleWrapper.classList.add("d-none");
  }
});

// Chuyển đổi giữa Ảnh gốc và Bản đồ nhiệt Grad-CAM
btnShowOriginal.addEventListener("click", () => {
  btnShowOriginal.classList.add("active");
  btnShowHeatmap.classList.remove("active");
  preview.classList.remove("d-none");
  heatmapPreview.classList.add("d-none");
});

btnShowHeatmap.addEventListener("click", () => {
  btnShowHeatmap.classList.add("active");
  btnShowOriginal.classList.remove("active");
  heatmapPreview.classList.remove("d-none");
  preview.classList.add("d-none");
});

// Nút xóa ảnh
resetBtn.addEventListener("click", (e) => {
  e.preventDefault();

  selectedFile = null;
  fileInput.value = "";
  preview.src = "";
  heatmapPreview.src = "";
  preview.classList.add("d-none");
  heatmapPreview.classList.add("d-none");
  uploadContent.classList.remove("d-none");
  heatmapToggleWrapper.classList.add("d-none");

  resultText.textContent = "Chưa có ảnh nào được phân tích.";
  resultText.className = "mb-1 fw-bold fs-5 text-dark";
  confidenceText.textContent = "";
  confidenceBar.style.width = "0%";
  scoreList.innerHTML = "";

  festivalInfo.classList.add("d-none");
  festivalTitle.textContent = "";
  festivalBadge.textContent = "";
  festivalDescription.textContent = "";
  festivalMeta.innerHTML = "";
  officialLinksContainer.classList.add("d-none");
  officialLinksList.innerHTML = "";
  modelsComparisonBox.classList.add("d-none");
  modelsList.innerHTML = "";
});

// Nút Nhận diện
predictBtn.addEventListener("click", async (e) => {
  e.preventDefault();

  if (!selectedFile) {
    resultText.textContent = "Vui lòng chọn một ảnh trước.";
    resultText.className = "mb-1 fw-bold fs-5 text-dark";
    festivalInfo.classList.add("d-none");
    return;
  }

  const formData = new FormData();
  formData.append("file", selectedFile);

  resultText.textContent = "Đang nhận diện...";
  resultText.className = "mb-1 fw-bold fs-5 text-dark";
  confidenceText.textContent = "";
  confidenceBar.style.width = "0%";
  scoreList.innerHTML = "";
  festivalInfo.classList.add("d-none");
  officialLinksContainer.classList.add("d-none");
  modelsComparisonBox.classList.add("d-none");
  modelsList.innerHTML = "";

  try {
    const res = await fetch(apiUrl, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      throw new Error("API error");
    }

    const data = await res.json();
    const confidencePct = data.confidence.toFixed(2);

    // Kiểm tra trường hợp ảnh ngoài phạm vi / không hợp lệ (OOD)
    if (data.is_ood || data.is_recognized === false) {
      resultText.textContent = data.name;
      resultText.className = "mb-1 fw-bold fs-5 text-danger";
      confidenceText.textContent = `Độ tin cậy quá thấp: ${confidencePct}% (Dưới ngưỡng quy định 55%)`;
      confidenceBar.style.width = "0%";
      confidenceBar.className = "progress-bar bg-danger";

      scoreList.innerHTML = "";
      festivalInfo.classList.add("d-none");
      officialLinksContainer.classList.add("d-none");
      heatmapToggleWrapper.classList.add("d-none");
      modelsComparisonBox.classList.add("d-none");
      return;
    }

    // Nếu ảnh hợp lệ (trong phạm vi):
    resultText.textContent = data.name;
    resultText.className = "mb-1 fw-bold fs-5 text-dark";
    confidenceText.textContent = `Độ tin cậy: ${confidencePct}%`;
    confidenceBar.style.width = `${confidencePct}%`;
    confidenceBar.className = "progress-bar bg-warning progress-bar-striped progress-bar-animated";

    // Hiển thị bảng Top 3
    if (data.top3 && data.top3.length > 0) {
      scoreList.innerHTML =
        '<div class="small fw-bold text-gold mb-1">XẾP HẠNG TOP 3 DỰ ĐOÁN:</div>';
      data.top3.forEach((item, index) => {
        const isTop1 = index === 0;
        const row = document.createElement("div");
        row.className = `score-row ${isTop1 ? "top-1" : ""}`;
        row.innerHTML = `
          <div class="d-flex justify-content-between align-items-center">
            <span class="small fw-semibold">#${index + 1} ${item.name}</span>
            <span class="small fw-bold ${isTop1 ? "text-gold" : "text-muted"}">${item.confidence.toFixed(1)}%</span>
          </div>
          <div class="mini-progress">
            <div class="mini-bar" style="width: ${item.confidence}%"></div>
          </div>
        `;
        scoreList.appendChild(row);
      });
    }

    // Hiển thị phần dự đoán 3 mô hình ở phía dưới ảnh (Cột bên trái, xếp từ cao xuống thấp)
    if (data.models_comparison) {
      const mc = data.models_comparison;
      bestBadge.textContent = `Tốt nhất: ${mc.best_model_name}`;
      evaluationMetricNote.textContent = "Đánh giá trên tập validation; confidence bên phải là của ảnh hiện tại.";
      const sortedModels = [...mc.models].sort(
        (a, b) => b.macro_f1 - a.macro_f1,
      );

      modelsList.innerHTML = "";
      sortedModels.forEach((m, idx) => {
        const isBest = idx === 0;
        const isLast = idx === sortedModels.length - 1;
        const li = document.createElement("li");
        li.className = `meta-item ${isLast ? "" : "mb-2"}`;
        li.innerHTML = `<strong>${m.name}:</strong> Accuracy ${m.accuracy.toFixed(2)}% · Top-3 ${m.top3_accuracy.toFixed(2)}% · Macro P/R/F1 ${m.macro_precision.toFixed(2)}% / ${m.macro_recall.toFixed(2)}% / ${m.macro_f1.toFixed(2)}% ${isBest ? '<span class="text-gold fw-bold ms-1">(Macro F1 cao nhất)</span>' : ""}`;
        modelsList.appendChild(li);
      });

      modelsComparisonBox.classList.remove("d-none");
    }

    // Hiển thị bản đồ nhiệt Grad-CAM nếu có
    if (data.heatmap) {
      heatmapPreview.src = data.heatmap;
      heatmapToggleWrapper.classList.remove("d-none");
      btnShowOriginal.classList.add("active");
      btnShowHeatmap.classList.remove("active");
      preview.classList.remove("d-none");
      heatmapPreview.classList.add("d-none");
    } else {
      heatmapToggleWrapper.classList.add("d-none");
    }

    // Hiển thị thông tin giới thiệu lễ hội
    festivalTitle.textContent = data.name;
    festivalBadge.textContent = data.label;
    festivalDescription.textContent = data.summary.description;
    festivalMeta.innerHTML = `
      <li class="meta-item mb-2"><strong>Thời gian:</strong> ${data.summary.time}</li>
      <li class="meta-item mb-2"><strong>Địa điểm:</strong> ${data.summary.place}</li>
      <li class="meta-item mb-2"><strong>Đối tượng:</strong> ${data.summary.people}</li>
      <li class="meta-item"><strong>Ý nghĩa:</strong> ${data.summary.meaning}</li>
    `;

    // Hiển thị các liên kết chính thống nếu có
    if (data.summary.official_links && data.summary.official_links.length > 0) {
      officialLinksList.innerHTML = "";
      data.summary.official_links.forEach((link) => {
        const a = document.createElement("a");
        a.className = "official-link-btn";
        a.href = link.url;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.textContent = `🔗 ${link.title}`;
        officialLinksList.appendChild(a);
      });
      officialLinksContainer.classList.remove("d-none");
    } else {
      officialLinksContainer.classList.add("d-none");
    }

    festivalInfo.classList.remove("d-none");
  } catch (error) {
    resultText.textContent = "Không thể kết nối tới API nhận diện.";
    confidenceText.textContent = "";
    confidenceBar.style.width = "0%";
  }
});
