build:
	sed -i 's/^version = .*/version = "$(v)"/' pyproject.toml
	sed -i 's/^version = .*/version = "$(v)"/' pytorch_eo/__init__.py
	poetry build

publish:
	poetry publish