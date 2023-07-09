all: build/V47.pdf

build/V47.pdf: V47.tex build lit.bib header.tex content/theorie.tex content/durchfuehrung.tex content/auswertung.tex content/diskussion.tex 
	lualatex --output-directory=build --interaction=batchmode --halt-on-error V47.tex
	biber build/V47.bcf
	lualatex --output-directory=build --interaction=batchmode --halt-on-error V47.tex

build : 
	mkdir -p build
clean : 
	rm -rf build

.PHONY : all clean
