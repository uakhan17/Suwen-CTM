# Data Preprocessing & Label Distribution Analysis

This directory contains the data cleaning scripts, preprocessing pipelines, and exploratory data analysis (EDA) for the dataset.

---

## 📊 Label Distribution Overview

Porfessors and graduate students from two Chinese medicine universities provided raw image data and defined 118 fine-grained labels for face and tongue diagnosis. Images and associated annotations can not be shared due to their proprietary nature. Face daignosis labels cover mouth, lip, eyes, nose and skin tones. Tongue diagnosis labels focus on tongue coating and tongue body. Below are the distribution bar charts for all target labels in the dataset. 

<details open>
<summary><b>Click to expand/collapse Label Distribution Charts (1 – 13)</b></summary>
<br>

<table>
  <tr>
    <td align="center"><b>面诊: 望口-口型</b></td>
    <td align="center"><b>面诊: 望口-口态</b></td>
  </tr>
  <tr>
    <td><img src="./images/face/面诊_望口_口形.png" width="400"/></td>
    <td><img src="./images/face/面诊_望口_口态.png" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><b>面诊: 望唇-唇形</b></td>
    <td align="center"><b>面诊: 望唇-唇色</b></td>
  </tr>
  <tr>
    <td><img src="./images/face/面诊_望唇_唇形.png" width="400"/></td>
    <td><img src="./images/face/面诊_望唇_唇色.png" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><b>面诊: 望眼-目形</b></td>
    <td align="center"><b>面诊: 望眼-目态</b></td>
  </tr>
  <tr>
    <td><img src="./images/face/面诊_望眼_目形.png" width="400"/></td>
    <td><img src="./images/face/面诊_望眼_目态.png" width="400"/></td>
  </tr>
   <tr>
    <td align="center"><b>面诊: 望眼-目色</b></td>
    <td align="center"><b>面诊: 望眼-瞳孔</b></td>
  </tr>
  <tr>
    <td><img src="./images/face/面诊_望眼_目色.png" width="400"/></td>
    <td><img src="./images/face/面诊_望眼_瞳孔.png" width="400"/></td>
  </tr>
   <tr>
    <td align="center"><b>面诊: 望鼻-鼻形</b></td>
    <td align="center"><b>面诊: 望鼻-鼻色</b></td>
  </tr>
  <tr>
    <td><img src="./images/face/面诊_望鼻_鼻形.png" width="400"/></td>
    <td><img src="./images/face/面诊_望鼻_鼻色.png" width="400"/></td>
  </tr>
   <tr>
    <td align="center"><b>面诊: 面色-皮肤光泽</b></td>
    <td align="center"><b>面诊: 面色-面形</b></td>
  </tr>
  <tr>
    <td><img src="./images/face/面诊_面色_皮肤光泽.png" width="400"/></td>
    <td><img src="./images/face/面诊_面色_面形.png" width="400"/></td>
  </tr>
   <tr>
    <td align="center"><b>面诊: 面色-面色</b></td>
  </tr>
  <tr>
    <td><img src="./images/face/面诊_面色_面色.png" width="400"/></td>
  </tr>
</table>

</details>

<details>
<summary><b>Click to expand/collapse Distribution Charts (14 – 19)</b></summary>
<br>

<table>
  <tr>
    <td align="center"><b>舌诊: 舌苔-苔色</b></td>
    <td align="center"><b>舌诊: 舌苔-苔质</b></td>
  </tr>
  <tr>
    <td><img src="./images/tongue/舌诊_舌苔_苔色.png" width="400"/></td>
    <td><img src="./images/tongue/舌诊_舌苔_苔质.png" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><b>舌诊: 舌质-形</b></td>
    <td align="center"><b>舌诊: 舌质-态</b></td>
  </tr>
  <tr>
    <td><img src="./images/tongue/舌诊_舌质_形.png" width="400"/></td>
    <td><img src="./images/tongue/舌诊_舌质_态.png" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><b>舌诊: 舌质-神</b></td>
    <td align="center"><b>舌诊: 舌质-色</b></td>
  </tr>
  <tr>
    <td><img src="./images/tongue/舌诊_舌质_神.png" width="400"/></td>
    <td><img src="./images/tongue/舌诊_舌质_色.png" width="400"/></td>
  </tr>
  <!-- Add additional rows using the same <tr>/<td> structure -->
</table>

</details>

---

## 📈 Key Insights & Imbalance Strategy

* **Class Imbalance:** The label distribution is severly biased, exhibiting a clear long-tail characteristic. Due to limited data availability, not all labels have sufficient data samples. 
* **Mitigation:** Experiment with different SOTA backbones and apply advanced data augmentation e.g., mixup, randaug and cutmix.