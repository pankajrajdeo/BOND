# Contributing to BOND

Thank you for your interest in contributing to BOND! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/pankajrajdeo/BOND/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Relevant error messages or logs

### Suggesting Features

1. Check if the feature has already been suggested
2. Open an issue with:
   - Clear description of the feature
   - Use case and motivation
   - Potential implementation approach (if you have ideas)

### Pull Requests

1. **Fork the repository** and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed
   - Keep commits focused and atomic

3. **Run tests and linting**:
   ```bash
   make test
   make lint
   ```

4. **Commit your changes**:
   ```bash
   git commit -m "Add: descriptive commit message"
   ```
   Follow conventional commit format when possible:
   - `Add:` for new features
   - `Fix:` for bug fixes
   - `Update:` for improvements
   - `Docs:` for documentation
   - `Refactor:` for code refactoring

5. **Push and create a PR**:
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a pull request on GitHub with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots/examples if applicable

## Development Setup

1. **Clone and setup**:
   ```bash
   git clone https://github.com/pankajrajdeo/BOND.git
   cd BOND
   python3.11 -m venv bond_venv
   source bond_venv/bin/activate
   pip install -e ".[dev]"
   ```

2. **Install pre-commit hooks** (optional):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Code Style

- Follow PEP 8 style guide
- Use type hints where possible
- Keep functions focused and small
- Add docstrings for public functions/classes
- Maximum line length: 100 characters (flexible for readability)

## Testing

- Write tests for new functionality
- Aim for high test coverage
- Run tests before submitting PR:
   ```bash
   pytest
   ```

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Update API documentation if endpoints change
- Keep examples in code current

## Project Structure

```
BOND/
├── bond/              # Main package
│   ├── pipeline.py   # Core BondMatcher class
│   ├── retrieval/    # Retrieval modules
│   ├── server.py     # FastAPI service
│   └── ...
├── scripts/          # Utility scripts
├── evals/            # Evaluation scripts
├── assets/           # Data files (gitignored)
└── tests/            # Test files
```

## Areas for Contribution

- **Performance improvements**: Optimization of retrieval or fusion
- **New ontology support**: Adding support for additional ontologies
- **Better error handling**: More informative error messages
- **Documentation**: Improving guides and examples
- **Tests**: Expanding test coverage
- **Benchmarking**: Evaluation on new datasets

## Questions?

Feel free to open an issue with the `question` label or reach out to the maintainers.

Thank you for contributing to BOND! 🎉

