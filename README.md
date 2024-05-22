# Linear Regression implementation in Go
An attempt to implement Linear Regression for N parameteres in Golang.


## Exposed Functions

### Fit()
```go
import (
    lr "github.com/omgupta1608/linear-regression-go"
)

func main() {
    model := lr.Fit(X, Y, nil, nil)
}
```

### Predict()
```go
import (
    lr "github.com/omgupta1608/linear-regression-go"
)

func main() {
    model := lr.Fit(X, Y, nil, nil)
    prediction := model.Predict(X_test)
}
```

### Access Coefficients and Intercept
```go
intercept := model._Intercept
coeffs := model._Coeffs
```
