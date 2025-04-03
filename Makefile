build:
	sed -i 's/^version = .*/version = "$(v)"/' pyproject.toml
	sed -i 's/^__version__ = .*/__version__ = "$(v)"/' pytorch_eo/__init__.py
	rm -rf dist
	uv build

publish:
	uv publish --username "__token__" --password "$(token)"