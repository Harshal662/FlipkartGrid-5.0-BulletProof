import React from "react";

// For styling
import { Box, Typography, makeStyles } from "@material-ui/core";

// CSS - Material UI
const useStyles = makeStyles((theme) => ({
  bTxt: {
    marginTop: "10px",
    marginBottom: "10px",
    backgroundColor: "red",
    height: "60px",
    textAlign: "center",
    width: "100%",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
  },
  boxTxt: {
    fontSize: "30px",
    fontWeight: "bold",
    color: "white",
    textAlign: "center",
    [theme.breakpoints.down("sm")]: {
      fontSize: "20px",
    },
  },
  txt: {
    textAlign: "center",
    fontSize: "30px",
    fontWeight: "bold",
    marginTop: "10px",
    marginBottom: "10px",
    [theme.breakpoints.down("sm")]: {
      fontSize: "20px",
    },
  },
}));

const Base = () => {
  const classes = useStyles();
  return (
    <>
      <Box className={classes.bTxt}>
        <Typography className={classes.boxTxt}>
          Flipkart Grid 5.0
        </Typography>
      </Box>
      <Typography className={classes.txt}>
        Fashion Recommendation System
      </Typography>
    </>
  );
};

export default Base;
