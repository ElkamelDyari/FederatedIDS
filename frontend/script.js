const form = document.getElementById("upload-form");
const outputDiv = document.getElementById("output");
const predictionsElement = document.getElementById("predictions");

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById("file");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select a file!");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("http://localhost:8080/predict", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Error: ${response.status}`);
        }

        const result = await response.json();

        if (result.error) {
            predictionsElement.textContent = `Error: ${result.error}`;
        } else {
            predictionsElement.textContent = JSON.stringify(result.predictions, null, 2);
        }

        outputDiv.classList.remove("hidden");
    } catch (error) {
        predictionsElement.textContent = `Error: ${error.message}`;
        outputDiv.classList.remove("hidden");
    }
});
