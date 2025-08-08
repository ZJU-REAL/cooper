<div align="center">

<h1 style="display: flex; justify-content: center; align-items: center; gap: 10px; margin: 0;">
Cooper: Co-Optimizing Policy and Reward Models in Reinforcement Learning for Large Language Models
</h1>
<p align="center"><em></em></p>

<p><em>A RL framework that jointly optimizes both the policy model and the reward model.</em></p>

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2508.05613) [![alphaXiv](https://img.shields.io/badge/discussion-A42C25?style=for-the-badge&logo=arxiv&logoColor=white&color=blue
)](https://www.alphaxiv.org/abs/2508.05613) [![Github](https://img.shields.io/badge/Cooper-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ZJU-REAL/Cooper)
</div>

<br>

<div align="center">
  <img src="./figures/cooper.png" alt="Cooper Framework" width="85%" />
  <p><em> An overview of the Cooper training framework. Each training step in Cooper consists of two stages: policy model optimization (blue area) and reward model optimization (green area).</em></p>
</div>

---

## 🎉 News
*   **[2025-8-7]** Our paper, **Cooper: Co-Optimizing Policy and Reward Models in Reinforcement Learning for Large Language Models**, is now available on arXiv!
*   **[Coming Soon]** We plan to release the code and dataset. Stay tuned!

---

## Table of Contents
* [Motivation](#motivation)
* [Highlights](#-highlights)
* [Citation](#-citation)
* [Acknowledgement](#-acknowledgement)

---

## Motivation
 Existing RL methods face a critical dilemma in reward design:

*   **Rule-based rewards** are precise but brittle. They struggle to parse diverse answer formats, leading to incorrect penalties that stifle model learning.
*   **Model-based rewards** (using a fixed reward model) are more robust but are vulnerable to **reward hacking**. The policy model can learn to exploit loopholes in the reward model, achieving high scores for incorrect answers and causing performance to collapse.

This forces a difficult choice between a reward system that is precise but inflexible, and one that is adaptable but easily exploited. How can we get the best of both worlds?

<div align="center">
  <img src="./figures/intro-cooper.png" alt="Overthinking Problem" style="width: 90%; height: auto;" />
</div>

This is where **Cooper** comes in. Cooper introduces a framework that **co-optimizes** both the policy and the reward model. It leverages the high precision of rule-based rewards to identify trustworthy positive samples, while an assistant LLM dynamically generates challenging negative samples.  This continuous stream of high-quality preference pairs is used to continuously refine the reward model, making it more robust and resistant to hacking. This dynamic process breaks the static reward dilemma, leading to more **stable** and **robust** RL training.


---
## ✨ Highlights

*   💡 **Co-Optimizing Framework**: Cooper is a novel framework to jointly and dynamically optimize both the policy and reward models during RL, breaking the limitations of static reward functions.
*   🛡️ **Mitigates Reward Hacking**: By continuously updating the reward model with high-quality data, Cooper effectively prevents the policy model from exploiting its weaknesses, ensuring stable and meaningful training.
*   ⚙️ **Dynamic Data Strategy**: Leverages a hybrid approach where high-precision rule-based rewards identify positive samples, and an assistant LLM generates challenging negative samples, constantly improving the reward model's accuracy.
*   🚀 **Improved Performance & Robustness**: Experiments show that Cooper not only alleviates reward hacking but also improves end-to-end performance, achieving a 3.09% gain in average accuracy on Qwen2.5-1.5B-Instruct.
---



## 🙏 Acknowledgement

Our RL training code is built upon the excellent [Verl](https://github.com/volcengine/verl) framework. We extend our sincere gratitude to their team for open-sourcing their powerful library.

---
## 📄 Citation

If you find Cooper useful in your research, please consider citing our work:

```bibtex
@misc{hong2025coopercooptimizingpolicyreward,
      title={Cooper: Co-Optimizing Policy and Reward Models in Reinforcement Learning for Large Language Models}, 
      author={Haitao Hong and Yuchen Yan and Xingyu Wu and Guiyang Hou and Wenqi Zhang and Weiming Lu and Yongliang Shen and Jun Xiao},
      year={2025},
      eprint={2508.05613},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.05613}, 
}

```
