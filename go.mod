module github.com/mattn/tensai

go 1.21

require github.com/ebitengine/purego v0.10.2

retract (
	v0.1.0 // Tagged in error before the package split; releases follow v0.0.x.
	v0.1.1 // Contains only this retraction.
)
