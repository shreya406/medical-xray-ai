const imageInput = document.getElementById("imageInput");
const analyzeButton = document.getElementById("analyzeButton");
const previewContainer = document.getElementById("previewContainer");
const previewImage = document.getElementById("previewImage");
const loading = document.getElementById("loading");
const results = document.getElementById("results");
const resultList = document.getElementById("resultList");

imageInput.addEventListener("change", function () {

    const file = imageInput.files[0];

    if (!file) {
        return;
    }

    previewImage.src = URL.createObjectURL(file);
    previewContainer.style.display = "block";
    results.style.display = "none";
});

analyzeButton.addEventListener("click", async function () {

    const file = imageInput.files[0];

    if (!file) {
        alert("Please select an X-ray image first.");
        return;
    }

    const formData = new FormData();

    formData.append("file", file);

    loading.style.display = "block";
    results.style.display = "none";

    try {

        const response = await fetch(
            "http://127.0.0.1:5000/predict",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed.");
        }

        resultList.innerHTML = "";

        for (const label in data) {

            const item = document.createElement("div");

            item.className = "result-item";

            item.innerHTML = `
                <span class="result-label">${label}</span>
                <span class="result-value">${data[label]}%</span>
            `;

            resultList.appendChild(item);
        }

        results.style.display = "block";

    } catch (error) {

        alert("Error: " + error.message);

    } finally {

        loading.style.display = "none";
    }
});