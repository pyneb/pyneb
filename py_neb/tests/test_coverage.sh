#Some things are cached in .coverage, which causes a problem if you change a test file name,
# for instance
rm .coverage

for f in $(find . -name "Test*.py"); do
    coverage run -a "$f" && coverage report
done

coverage html
