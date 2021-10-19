for f in $(find . -name "Test*.py"); do
    coverage run -a "$f" && coverage report
done

coverage html
