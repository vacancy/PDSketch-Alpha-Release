# PDSketch-Release

**PDSketch: Integrated Domain Programming, Learning, and Planning**

**Warning**: This is a very alpha release of the code. It contains basic code for learning and planning to reproduce results described in the original NeurIPS 2022 paper. After the publication of the paper, we have been actively working on an improved version of the code that also contains integration with other neuro-symbolic methods and more sophiscated task and motion planning for robots. The release date for the new version is scheduled at summer 2023. Please only use this code as a reference.

---

<div align="center">
  <img src="https://pdsketch.csail.mit.edu/data/img/teaser.png" width="100%">
</div>

**[PDSketch: Integrated Domain Programming, Learning, and Planning](https://pdsketch.csail.mit.edu/data/papers/2022NeurIPS-PDSketch.pdf)**
<br />
[Jiayuan Mao](http://jiayuanm.com), 
[Tomás Lozano-Pérez](https://people.csail.mit.edu/tlp/),
[Joshua B. Tenenbaum](https://web.mit.edu/cocosci/josh.html), and
[Leslie Pack Kaelbling](https://people.csail.mit.edu/lpk/)
<br />
In Conference on Neural Information Processing Systems (NeurIPS) 2022
<br />
[[Paper]](https://pdsketch.csail.mit.edu/data/papers/2022NeurIPS-PDSketch.pdf)
[[Supplementary]](https://pdsketch.csail.mit.edu/data/papers/2022NeurIPS-PDSketch-Supp.pdf)
[[Project Page]](https://pdsketch.csail.mit.edu/)
[[BibTex]](https://pdsketch.csail.mit.edu/data/bibtex/2022NeurIPS-PDSketch.bib)

```
@inproceedings{Mao2022PDSketch,
  title={{PDSketch: Integrated Domain Programming, Learning, and Planning}},
  author={Mao, Jiayuan and Lozano-Perez, Tomas and Tenenbaum, Joshua B. and Leslie Pack Kaelbing},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```

## Prerequisites

- Python 3
- PyTorch 1.10 or higher, with NVIDIA CUDA Support

## Installation

Install [Jacinle](https://github.com/vacancy/Jacinle): Clone the package, and add the bin path to your global `PATH` environment variable:

```
git clone https://github.com/vacancy/Jacinle --recursive
export PATH=<path_to_jacinle>/bin:$PATH
```

## Minigrid Experiments

```
./expr.sh basic
./eval.sh basic  # the last two commands will fail because the basic model does not support ao_discretization

./expr.sh abskin
./eval.sh abskin

./expr.sh full
./eval.sh full
```
