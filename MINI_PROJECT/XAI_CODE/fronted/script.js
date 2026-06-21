// const API_URL = 'http://127.0.0.1:8000/predict'; // Change if deployed

// async function predict() {
//     const fileInput = document.getElementById('imageUpload');
//     if (!fileInput.files.length) {
//         alert('Please select an image.');
//         return;
//     }

//     const method = document.getElementById('methodSelect').value;
//     const formData = new FormData();
//     formData.append('file', fileInput.files[0]);

//     // Show loader, hide previous results
//     document.getElementById('loader').style.display = 'block';
//     document.getElementById('results').style.display = 'none';

//     try {
//         const response = await fetch(`${API_URL}?method=${method}`, {
//             method: 'POST',
//             body: formData
//         });
//         const data = await response.json();

//         if (data.success) {
//             // Display images
//             document.getElementById('originalImg').src = 'data:image/png;base64,' + data.original_image;
//             document.getElementById('heatmapImg').src = 'data:image/png;base64,' + data.heatmap;

//             // Display prediction
//             const pred = data.prediction;
//             document.getElementById('predictionText').innerText = `Prediction: ${pred.predicted_class}`;
//             document.getElementById('confidenceText').innerText = `Confidence: ${(pred.confidence * 100).toFixed(2)}%`;

//             // Class probabilities
//             const probList = document.getElementById('probList');
//             probList.innerHTML = '';
//             for (const [cls, prob] of Object.entries(pred.probabilities)) {
//                 const li = document.createElement('li');
//                 li.innerText = `${cls}: ${(prob * 100).toFixed(2)}%`;
//                 probList.appendChild(li);
//             }

//             // Region scores
//             const regionList = document.getElementById('regionList');
//             regionList.innerHTML = '';
//             if (data.region_scores && Object.keys(data.region_scores).length > 0) {
//                 for (const [region, score] of Object.entries(data.region_scores)) {
//                     const li = document.createElement('li');
//                     li.innerText = `${region.replace(/_/g, ' ')}: ${(score * 100).toFixed(2)}%`;
//                     regionList.appendChild(li);
//                 }
//             } else {
//                 regionList.innerHTML = '<li>No region scores available</li>';
//             }

//             document.getElementById('results').style.display = 'block';
//         } else {
//             alert('Error: ' + data.error);
//         }
//     } catch (err) {
//         console.error(err);
//         alert('Failed to connect to backend.');
//     } finally {
//         document.getElementById('loader').style.display = 'none';
//     }
//     async function predictProgression() {
//     const form = document.getElementById('progressionForm');
//     const formData = new FormData(form);
//     const data = {};
//     formData.forEach((value, key) => {
//         data[key] = value === '' ? null : parseFloat(value);
//     });

//     document.getElementById('progressionResult').style.display = 'none';

//     try {
//         const response = await fetch('http://127.0.0.1:8000/predict_progression', {
//             method: 'POST',
//             headers: { 'Content-Type': 'application/json' },
//             body: JSON.stringify(data)
//         });
//         const result = await response.json();

//         if (result.success) {
//             document.getElementById('hazardRatio').innerText = `Hazard Ratio: ${result.hazard_ratio.toFixed(3)}`;
//             let interpretation = result.hazard_ratio > 1 ? 'Higher risk of progression' : 'Lower risk of progression';
//             document.getElementById('riskInterpretation').innerText = interpretation;
//             document.getElementById('progressionResult').style.display = 'block';
//         } else {
//             alert('Error: ' + result.error);
//         }
//     } catch (err) {
//         console.error(err);
//         alert('Failed to connect to backend.');
//     }
// }
// }
// // async function predictLSTM() {
// //     const files = document.getElementById('lstmFiles').files;
// //     if (files.length === 0) {
// //         alert('Please select at least one image.');
// //         return;
// //     }
// //     const formData = new FormData();
// //     for (let i = 0; i < files.length; i++) {
// //         formData.append('files', files[i]);
// //     }
// //     try {
// //         const response = await fetch('http://127.0.0.1:8000/predict_progression_lstm', {
// //             method: 'POST',
// //             body: formData
// //         });
// //         const data = await response.json();
// //         if (data.success) {
// //             document.getElementById('lstmResult').innerHTML = `
// //                 <h3>Predicted Future CDR: ${data.predicted_cdr}</h3>
// //                 <p>Based on ${data.num_visits_used} past visits (max allowed: ${data.max_len}).</p>
// //             `;
// //         } else {
// //             alert('Error: ' + data.error);
// //         }
// //     } catch (err) {
// //         alert('Failed to connect.');
// //     }
// // }



// // async function predictLSTM() {
// //     const files = document.getElementById('lstmFiles').files;
// //     if (files.length === 0) {
// //         alert('Please select at least one image.');
// //         return;
// //     }
// //     const formData = new FormData();
// //     for (let i = 0; i < files.length; i++) {
// //         formData.append('files', files[i]);
// //     }
// //     try {
// //         const response = await fetch('http://127.0.0.1:8000/predict_progression_lstm', {
// //             method: 'POST',
// //             body: formData
// //         });
// //         const data = await response.json();
// //         if (data.success) {
// //             document.getElementById('lstmResult').innerHTML = `
// //                 <h3>Predicted Future CDR: ${data.predicted_cdr}</h3>
// //                 <p>Based on ${data.num_visits_used} past visits (max allowed: ${data.max_len}).</p>
// //             `;
// //         } else {
// //             alert('Error: ' + data.error);
// //         }
// //     } catch (err) {
// //         alert('Failed to connect.');
// //     }
// // }


// // async function predictLSTM() {
// //     const files = document.getElementById('lstmFiles').files;
// //     if (files.length === 0) {
// //         alert('Please select at least one image.');
// //         return;
// //     }
// //     const formData = new FormData();
// //     for (let i = 0; i < files.length; i++) {
// //         formData.append('files', files[i]);
// //     }
// //     try {
// //         const response = await fetch('http://127.0.0.1:8000/predict_progression_lstm', {
// //             method: 'POST',
// //             body: formData
// //         });
// //         const data = await response.json();
// //         if (data.success) {
// //             // Determine progression rate based on predicted CDR
// //             let rateMessage = '';
// //             if (data.predicted_cdr < 0.25) {
// //                 rateMessage = '🟢 Slow progression (likely stable)';
// //             } else if (data.predicted_cdr < 0.75) {
// //                 rateMessage = '🟡 Moderate progression (towards very mild dementia)';
// //             } else {
// //                 rateMessage = '🔴 Fast progression (significant decline)';
// //             }

// //             document.getElementById('lstmResult').innerHTML = `
// //                 <h3>Predicted Future CDR: ${data.predicted_cdr}</h3>
// //                 <p><strong>Progression Rate:</strong> ${rateMessage}</p>
// //                 <p>Based on ${data.num_visits_used} past visits (max allowed: ${data.max_len}).</p>
// //             `;
// //         } else {
// //             alert('Error: ' + data.error);
// //         }
// //     } catch (err) {
// //         alert('Failed to connect.');
// //     }
// // }

// // ===== Helper: update file name display =====
// document.addEventListener('DOMContentLoaded', function() {
//     document.getElementById('imageUpload')?.addEventListener('change', function(e) {
//         document.getElementById('file-name').textContent = e.target.files[0]?.name || 'No file chosen';
//     });
//     document.getElementById('lstmFiles')?.addEventListener('change', function(e) {
//         const count = e.target.files.length;
//         document.getElementById('lstm-file-count').textContent = count ? count + ' files selected' : 'No files chosen';
//     });
// });

// // ===== 1. Classification =====
// async function predict() {
//     console.log("predict() called"); // debug
//     const fileInput = document.getElementById('imageUpload');
//     if (!fileInput || !fileInput.files.length) {
//         alert('Please select an image.');
//         return;
//     }

//     const method = document.getElementById('methodSelect').value;
//     const formData = new FormData();
//     formData.append('file', fileInput.files[0]);

//     document.getElementById('loader').style.display = 'block';
//     document.getElementById('results').style.display = 'none';

//     try {
//         const response = await fetch(`${API_URL}?method=${method}`, {
//             method: 'POST',
//             body: formData
//         });
//         const data = await response.json();

//         if (data.success) {
//             document.getElementById('originalImg').src = 'data:image/png;base64,' + data.original_image;
//             document.getElementById('heatmapImg').src = 'data:image/png;base64,' + data.heatmap;

//             const pred = data.prediction;
//             document.getElementById('predictionText').innerText = `Prediction: ${pred.predicted_class}`;
//             document.getElementById('confidenceText').innerText = `Confidence: ${(pred.confidence * 100).toFixed(2)}%`;

//             const probList = document.getElementById('probList');
//             probList.innerHTML = '';
//             for (const [cls, prob] of Object.entries(pred.probabilities)) {
//                 const li = document.createElement('li');
//                 li.innerText = `${cls}: ${(prob * 100).toFixed(2)}%`;
//                 probList.appendChild(li);
//             }

//             const regionList = document.getElementById('regionList');
//             regionList.innerHTML = '';
//             if (data.region_scores && Object.keys(data.region_scores).length > 0) {
//                 for (const [region, score] of Object.entries(data.region_scores)) {
//                     const li = document.createElement('li');
//                     li.innerText = `${region.replace(/_/g, ' ')}: ${(score * 100).toFixed(2)}%`;
//                     regionList.appendChild(li);
//                 }
//             } else {
//                 regionList.innerHTML = '<li>No region scores available</li>';
//             }

//             document.getElementById('results').style.display = 'block';
//         } else {
//             alert('Error: ' + data.error);
//         }
//     } catch (err) {
//         console.error('Fetch error:', err);
//         alert('Failed to connect to backend. Is the server running on port 8000?');
//     } finally {
//         document.getElementById('loader').style.display = 'none';
//     }
// }

// // ===== 2. Clinical Progression (Cox) =====
// async function predictProgression() {
//     const form = document.getElementById('progressionForm');
//     const formData = new FormData(form);
//     const data = {};
//     formData.forEach((value, key) => {
//         data[key] = value === '' ? null : parseFloat(value);
//     });

//     document.getElementById('progressionResult').style.display = 'none';

//     try {
//         const response = await fetch('http://127.0.0.1:8000/predict_progression', {
//             method: 'POST',
//             headers: { 'Content-Type': 'application/json' },
//             body: JSON.stringify(data)
//         });
//         const result = await response.json();

//         if (result.success) {
//             document.getElementById('hazardRatio').innerText = result.hazard_ratio.toFixed(3);
//             const interpretation = result.hazard_ratio > 1
//                 ? '🔴 Higher risk of progression'
//                 : '🟢 Lower risk of progression';
//             document.getElementById('riskInterpretation').innerText = interpretation;
//             document.getElementById('progressionResult').style.display = 'block';
//         } else {
//             alert('Error: ' + result.error);
//         }
//     } catch (err) {
//         console.error('Progression fetch error:', err);
//         alert('Failed to connect to backend.');
//     }
// }

// // ===== 3. MRI‑Based Progression (LSTM) =====
// async function predictLSTM() {
//     const files = document.getElementById('lstmFiles').files;
//     if (files.length === 0) {
//         alert('Please select at least one image.');
//         return;
//     }

//     const formData = new FormData();
//     for (let i = 0; i < files.length; i++) {
//         formData.append('files', files[i]);
//     }

//     try {
//         const response = await fetch('http://127.0.0.1:8000/predict_progression_lstm', {
//             method: 'POST',
//             body: formData
//         });
//         const data = await response.json();

//         if (data.success) {
//             let rateMessage = '';
//             if (data.predicted_cdr < 0.25) {
//                 rateMessage = '🟢 Slow progression (likely stable)';
//             } else if (data.predicted_cdr < 0.75) {
//                 rateMessage = '🟡 Moderate progression (towards very mild dementia)';
//             } else {
//                 rateMessage = '🔴 Fast progression (significant decline)';
//             }

//             document.getElementById('predictedCDR').innerText = data.predicted_cdr;
//             document.getElementById('lstmInterpretation').innerText = rateMessage;
//             document.getElementById('lstmResult').style.display = 'block';
//         } else {
//             alert('Error: ' + data.error);
//         }
//     } catch (err) {
//         console.error('LSTM fetch error:', err);
//         alert('Failed to connect.');
//     }
// }
// ===== API Endpoints =====
const API_URL = 'http://127.0.0.1:8000/predict';

// ===== DOMContentLoaded: attach file input listeners =====
document.addEventListener('DOMContentLoaded', function() {
    // Classification
    document.getElementById('imageUpload')?.addEventListener('change', function(e) {
        document.getElementById('file-name').textContent = e.target.files[0]?.name || 'No file chosen';
    });
    // LSTM multiple files
    document.getElementById('lstmFiles')?.addEventListener('change', function(e) {
        const count = e.target.files.length;
        document.getElementById('lstm-file-count').textContent = count ? count + ' files selected' : 'No files chosen';
    });
    // XGBoost single file
    document.getElementById('xgbImage')?.addEventListener('change', function(e) {
        document.getElementById('xgb-file-name').textContent = e.target.files[0]?.name || 'No file chosen';
    });
});

// ===== 1. Classification =====
async function predict() {
    const fileInput = document.getElementById('imageUpload');
    if (!fileInput.files.length) {
        alert('Please select an image.');
        return;
    }

    const method = document.getElementById('methodSelect').value;
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    document.getElementById('loader').style.display = 'block';
    document.getElementById('results').style.display = 'none';

    try {
        const response = await fetch(`${API_URL}?method=${method}`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.success) {
            document.getElementById('originalImg').src = 'data:image/png;base64,' + data.original_image;
            document.getElementById('heatmapImg').src = 'data:image/png;base64,' + data.heatmap;

            const pred = data.prediction;
            document.getElementById('predictionText').innerText = `Prediction: ${pred.predicted_class}`;
            document.getElementById('confidenceText').innerText = `Confidence: ${(pred.confidence * 100).toFixed(2)}%`;

            const probList = document.getElementById('probList');
            probList.innerHTML = '';
            for (const [cls, prob] of Object.entries(pred.probabilities)) {
                const li = document.createElement('li');
                li.innerText = `${cls}: ${(prob * 100).toFixed(2)}%`;
                probList.appendChild(li);
            }

            const regionList = document.getElementById('regionList');
            regionList.innerHTML = '';
            if (data.region_scores && Object.keys(data.region_scores).length > 0) {
                for (const [region, score] of Object.entries(data.region_scores)) {
                    const li = document.createElement('li');
                    li.innerText = `${region.replace(/_/g, ' ')}: ${(score * 100).toFixed(2)}%`;
                    regionList.appendChild(li);
                }
            } else {
                regionList.innerHTML = '<li>No region scores available</li>';
            }

            document.getElementById('results').style.display = 'block';
        } else {
            alert('Error: ' + data.error);
        }
    } catch (err) {
        console.error(err);
        alert('Failed to connect to backend.');
    } finally {
        document.getElementById('loader').style.display = 'none';
    }
}

// ===== 2. Clinical Progression (Cox) =====
async function predictProgression() {
    const form = document.getElementById('progressionForm');
    const formData = new FormData(form);
    const data = {};
    formData.forEach((value, key) => {
        data[key] = value === '' ? null : parseFloat(value);
    });

    document.getElementById('progressionResult').style.display = 'none';

    try {
        const response = await fetch('http://127.0.0.1:8000/predict_progression', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await response.json();

        if (result.success) {
            document.getElementById('hazardRatio').innerText = result.hazard_ratio.toFixed(3);
            const interpretation = result.hazard_ratio > 1
                ? '🔴 Higher risk of progression'
                : '🟢 Lower risk of progression';
            document.getElementById('riskInterpretation').innerText = interpretation;
            document.getElementById('progressionResult').style.display = 'block';
        } else {
            alert('Error: ' + result.error);
        }
    } catch (err) {
        console.error(err);
        alert('Failed to connect to backend.');
    }
}

// ===== 3. MRI‑Based Progression (LSTM) =====
async function predictLSTM() {
    const files = document.getElementById('lstmFiles').files;
    if (files.length === 0) {
        alert('Please select at least one image.');
        return;
    }

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }

    try {
        const response = await fetch('http://127.0.0.1:8000/predict_progression_lstm', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.success) {
            let rateMessage = '';
            if (data.predicted_cdr < 0.25) {
                rateMessage = '🟢 Slow progression (likely stable)';
            } else if (data.predicted_cdr < 0.75) {
                rateMessage = '🟡 Moderate progression (towards very mild dementia)';
            } else {
                rateMessage = '🔴 Fast progression (significant decline)';
            }

            document.getElementById('predictedCDR').innerText = data.predicted_cdr;
            document.getElementById('lstmInterpretation').innerText = rateMessage;
            document.getElementById('lstmResult').style.display = 'block';
        } else {
            alert('Error: ' + data.error);
        }
    } catch (err) {
        console.error(err);
        alert('Failed to connect.');
    }
}

// ===== 4. XGBoost MMSE Progression =====
async function predictXGB() {
    const fileInput = document.getElementById('xgbImage');
    if (!fileInput.files.length) {
        alert('Please select an MRI image.');
        return;
    }
    const form = document.getElementById('xgbForm');
    const formData = new FormData(form);
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch('http://127.0.0.1:8000/predict_progression_xgb', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (data.success) {
            document.getElementById('predictedMMSE').innerText = data.predicted_next_mmse;
            document.getElementById('xgbInterpretation').innerHTML = `
                <p>Next MMSE predicted: ${data.predicted_next_mmse}</p>
                <p>Change: ${data.mmse_change} points</p>
                <p>Progression rate: ${data.progression_rate}</p>
                <p class="note">(Negative change = decline)</p>
            `;
            document.getElementById('xgbResult').style.display = 'block';
        } else {
            alert('Error: ' + data.error);
        }
    } catch (err) {
        console.error(err);
        alert('Failed to connect.');
    }
}