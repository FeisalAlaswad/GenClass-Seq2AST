# Neural Translation of Natural Language to UML Class Diagrams

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Transformer-based approach for automated generation of UML class diagrams from natural language requirements.

## Key Features
- **Seq2Seq Model**: Direct PlantUML code generation
- **Seq2AST Model**: Syntax-valid diagram generation via Abstract Syntax Trees
- **Multi-Task Learning**: Joint optimization for:
  - Element identification
  - Design pattern suggestion
- **Structural Enhancements**: Incorporates frequent design patterns from 172k+ existing diagrams

## Performance
| Metric         | Seq2Seq | Seq2AST (Enhanced) |
|----------------|---------|--------------------|
| BLEU-4         | 38.67   | 56.38              |
| Recall (%)     | -   | 92.31              |
| Precision (%)  | -   | 87.64              |
| F1-Score (%)   | -   | 89.90              |

## Installation
```bash
git clone https://github.com/FeisalAlaswad/GenClass-Seq2AST.git
cd GenClass-Seq2AST
pip install -r requirements.txt
```



## Datasets
- **PlantUCD**: 3,777 annotated requirements
  Available at: [Dataset Repo](https://github.com/FeisalAlaswad/PlantUCD-dataset-full)
- **ModelSet**: 172,682 structured class diagrams  
  Available at: [Dataset Repo](https://github.com/FeisalAlaswad/ModelSet-AST-Structured-JSON)



## Citation
If you use this work or dataset in academic research, please cite it as follows:
Not yet....
```bibtex
@article{alaswad2025neural,
  title={Automating UML Class Diagram Generation from Natural Language via Transformer-Based Structured Translation},
  author={Alaswad, Feisal and Eswaran, Poovammal},
  journal={-},
  volume={-},
  number={-},
  year={2025}
}
```

## License
MIT License

Copyright (c) 2025 Feisal Alaswad

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
